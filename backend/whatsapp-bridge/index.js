/**
 * RAKSHA AI — WhatsApp Bridge (Baileys)
 *
 * Uses @whiskeysockets/baileys to connect directly to WhatsApp's WebSocket.
 * Exposes REST API + WebSocket for the FastAPI backend.
 */

// Baileys is ESM-only in current releases.  Keep this bridge CommonJS for its
// existing Express setup, then load Baileys with dynamic import before opening
// the server.  `require()` crashes the entire Render sidecar with ERR_REQUIRE_ESM.
let makeWASocket;
let useMultiFileAuthState;
let DisconnectReason;
let fetchLatestBaileysVersion;
let downloadMediaMessage;
let getContentType;
let isJidGroup;
let isJidUser;
let isJidBroadcast;
let isJidNewsletter;
const QRCode   = require("qrcode");
const express  = require("express");
const cors     = require("cors");
const { WebSocketServer } = require("ws");
const http     = require("http");
const path     = require("path");
const fs       = require("fs");
const pino     = require("pino");

try { require("dotenv").config({ path: path.join(__dirname, ".env") }); } catch {}

const PORT            = parseInt(process.env.PORT || "3001", 10);
const AUTH_DIR        = path.join(__dirname, ".baileys_auth");
const MAX_CHATS       = parseInt(process.env.MAX_CHATS || "5000", 10);
const MAX_MSGS_STORED = parseInt(process.env.MAX_MSGS_STORED || "250", 10);
const MAX_MEDIA_BYTES = parseInt(process.env.MAX_MEDIA_BYTES || `${2 * 1024 * 1024}`, 10);
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "http://localhost:3000,http://localhost:8000")
  .split(",").map((o) => o.trim());

const logger = pino({ level: "warn" });

// ── Express + WebSocket ────────────────────────────────────────────────────
const app    = express();
app.use(cors({ origin: ALLOWED_ORIGINS, credentials: true }));
app.use(express.json());
const server = http.createServer(app);
const wss    = new WebSocketServer({ server, path: "/ws" });
const wsClients = new Set();

wss.on("connection", (ws) => {
  wsClients.add(ws);
  ws.send(JSON.stringify({
    event: "status",
    data: { connected: clientReady, qrReady: !!lastQrDataUrl, phone: connectedPhone },
  }));
  if (lastQrDataUrl && !clientReady) {
    ws.send(JSON.stringify({ event: "qr", data: { qr: lastQrDataUrl } }));
  }
  ws.on("close", () => wsClients.delete(ws));
});

function broadcast(event, data) {
  const payload = JSON.stringify({ event, data });
  for (const ws of wsClients) if (ws.readyState === 1) ws.send(payload);
}

// ── Global state ───────────────────────────────────────────────────────────
let clientReady    = false;
let connectedPhone = null;
let lastQrDataUrl  = null;
let sock           = null;
let chatStore      = {};   // jid → chat metadata
let messageStore   = {};   // jid → Message[]
let syncRetryTimer = null;
let reconnectTimer = null;

// ── JID helpers ────────────────────────────────────────────────────────────
function isRealChat(jid) {
  if (!jid) return false;
  if (isJidBroadcast(jid))   return false;
  if (jid === "status@broadcast") return false;
  return isJidUser(jid) || isJidGroup(jid) || isJidNewsletter(jid) || jid.endsWith("@lid");
}

function chatKind(jid, chat = {}) {
  if (isJidNewsletter(jid)) return "channel";
  if (isJidGroup(jid)) {
    if (chat.isCommunity || chat.linkedParent || chat.parentGroupJid) return "community";
    return "group";
  }
  return "personal";
}

function unwrapMessage(msg) {
  let current = msg || {};
  for (let i = 0; i < 5; i++) {
    if (current.ephemeralMessage?.message) current = current.ephemeralMessage.message;
    else if (current.viewOnceMessage?.message) current = current.viewOnceMessage.message;
    else if (current.viewOnceMessageV2?.message) current = current.viewOnceMessageV2.message;
    else if (current.documentWithCaptionMessage?.message) current = current.documentWithCaptionMessage.message;
    else break;
  }
  return current;
}

function isUserVisibleMessage(msg) {
  const type = getContentType(unwrapMessage(msg));
  return !!type && ![
    "protocolMessage",
    "senderKeyDistributionMessage",
    "historySyncNotification",
    "appStateSyncKeyShare",
    "messageContextInfo",
  ].includes(type);
}

// ── Extract text from any message type ────────────────────────────────────
function extractBody(msg) {
  const m = unwrapMessage(msg);
  if (!m) return "";
  return (
    m.conversation                                          ||
    m.extendedTextMessage?.text                            ||
    m.imageMessage?.caption                                ||
    m.videoMessage?.caption                                ||
    m.documentMessage?.caption                             ||
    m.buttonsMessage?.contentText                          ||
    m.buttonsResponseMessage?.selectedDisplayText          ||
    m.listMessage?.description                             ||
    m.listResponseMessage?.title                           ||
    m.templateMessage?.hydratedTemplate?.hydratedContentText ||
    m.pollCreationMessage?.name                            ||
    m.pollUpdateMessage?.name                              ||
    ""
  );
}

function extractMedia(msg) {
  const m = unwrapMessage(msg);
  const contentType = getContentType(m) || "unknown";
  const mediaNode =
    m.imageMessage || m.videoMessage || m.audioMessage || m.documentMessage ||
    m.stickerMessage || m.ptvMessage || null;

  if (mediaNode) {
    const kind = contentType.replace("Message", "").replace("ptv", "video");
    return {
      kind,
      contentType,
      mimeType: mediaNode.mimetype || mediaNode.mimeType || null,
      fileName: mediaNode.fileName || null,
      caption: mediaNode.caption || "",
      fileLength: Number(mediaNode.fileLength?.toString?.() || mediaNode.fileLength || 0) || null,
      seconds: mediaNode.seconds || null,
      width: mediaNode.width || null,
      height: mediaNode.height || null,
      hasPreview: !!mediaNode.jpegThumbnail,
      dataUrl: null,
    };
  }

  if (m.locationMessage || m.liveLocationMessage) {
    const loc = m.locationMessage || m.liveLocationMessage;
    return {
      kind: "location",
      contentType,
      mimeType: null,
      caption: loc.name || loc.address || "",
      latitude: loc.degreesLatitude,
      longitude: loc.degreesLongitude,
      dataUrl: null,
    };
  }

  if (m.contactMessage || m.contactsArrayMessage) {
    return {
      kind: "contact",
      contentType,
      mimeType: null,
      caption: m.contactMessage?.displayName || `${m.contactsArrayMessage?.contacts?.length || 0} contacts`,
      dataUrl: null,
    };
  }

  return {
    kind: contentType === "conversation" || contentType === "extendedTextMessage" ? "text" : contentType.replace("Message", ""),
    contentType,
    mimeType: null,
    caption: "",
    dataUrl: null,
  };
}

function messagePreview(body, media) {
  if (body) return body;
  if (!media || media.kind === "text") return "";
  if (media.kind === "location") return media.caption ? `Location: ${media.caption}` : "Location shared";
  if (media.kind === "contact") return media.caption ? `Contact: ${media.caption}` : "Contact shared";
  if (media.fileName) return `${media.kind}: ${media.fileName}`;
  return `${media.kind} message`;
}

function shouldInlineMedia(media) {
  if (!media) return false;
  if (!["image", "sticker", "video", "audio", "document"].includes(media.kind)) return false;
  if (media.fileLength && media.fileLength > MAX_MEDIA_BYTES) return false;
  return ["image", "sticker"].includes(media.kind) || (media.fileLength && media.fileLength <= MAX_MEDIA_BYTES);
}

async function maybeAttachMedia(raw, media) {
  if (!sock || !shouldInlineMedia(media)) return media;
  try {
    const buffer = await downloadMediaMessage(
      raw,
      "buffer",
      {},
      { logger, reuploadRequest: sock.updateMediaMessage }
    );
    if (!buffer || buffer.length > MAX_MEDIA_BYTES) return media;
    const mime = media.mimeType || "application/octet-stream";
    return { ...media, dataUrl: `data:${mime};base64,${buffer.toString("base64")}` };
  } catch (err) {
    return { ...media, downloadError: err.message?.slice(0, 120) || "media unavailable" };
  }
}

// ── Upsert helpers ─────────────────────────────────────────────────────────
function upsertChat(chat) {
  if (!chat?.id || !isRealChat(chat.id)) return;
  chatStore[chat.id] = { ...(chatStore[chat.id] || {}), ...chat };
}

async function upsertMessage(raw) {
  if (!raw?.message || !isUserVisibleMessage(raw.message)) return;
  const body   = extractBody(raw.message);
  const media  = await maybeAttachMedia(raw, extractMedia(raw.message));
  const chatId = raw.key.remoteJid || "";
  if (!isRealChat(chatId)) return;

  if (!messageStore[chatId]) messageStore[chatId] = [];
  const arr = messageStore[chatId];

  if (arr.some((m) => m.id === raw.key.id)) return; // dedup

  const timestamp = typeof raw.messageTimestamp === "number"
    ? raw.messageTimestamp
    : parseInt(raw.messageTimestamp?.toString() || "0");

  arr.push({
    id:        raw.key.id,
    from:      chatId,
    author:    raw.key.participant || (raw.key.fromMe ? "me" : chatId),
    body,
    preview:   messagePreview(body, media),
    timestamp,
    fromMe:    raw.key.fromMe || false,
    type:      media.kind || "text",
    media,
  });

  const existingChat = chatStore[chatId] || {};
  upsertChat({
    id: chatId,
    name: existingChat.name || existingChat.subject || chatId.split("@")[0],
    conversationTimestamp: Math.max(timestamp || 0, existingChat.conversationTimestamp || 0),
    unreadCount: existingChat.unreadCount || 0,
  });

  // Keep only the MAX_MSGS_STORED most recent
  if (arr.length > MAX_MSGS_STORED * 2) {
    arr.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
    arr.splice(MAX_MSGS_STORED);
  }
}

function getMessages(chatId, limit = MAX_MSGS_STORED) {
  const arr = messageStore[chatId] || [];
  return [...arr]
    .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))
    .slice(0, Math.min(limit, MAX_MSGS_STORED));
}

function lastStoredMessage(chatId) {
  return getMessages(chatId, 1)[0] || null;
}

function logStats(label = "") {
  const all      = Object.values(chatStore);
  const personal = all.filter((c) => isJidUser(c.id)).length;
  const groups   = all.filter((c) => isJidGroup(c.id)).length;
  const msgs     = Object.values(messageStore).reduce((s, a) => s + a.length, 0);
  console.log(`[WA]${label ? " " + label + " |" : ""} Store: ${all.length} chats (${personal} personal, ${groups} groups), ${msgs} messages`);
}

function scheduleReconnect(activeSocket, delayMs) {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    if (sock === activeSocket) startWhatsApp();
  }, delayMs);
}

// ── Attempt AppState resync to recover chats without full re-auth ──────────
async function trySyncChats(attempt = 1) {
  if (!clientReady || !sock) return;
  const chatCount = Object.keys(chatStore).filter(
    (id) => isJidUser(chatStore[id]?.id)
  ).length;

  if (chatCount > 0) {
    console.log(`[WA] Already have ${chatCount} personal chats — skip resync`);
    return;
  }

  console.log(`[WA] Personal chats missing, requesting AppState resync (attempt ${attempt}/3)...`);
  try {
    await sock.resyncAppState(
      ["critical_block", "critical_unblock_low", "regular_high", "regular_low"],
      false
    );
    console.log("[WA] AppState resync requested — waiting for chats.set event...");
  } catch (err) {
    console.error("[WA] resyncAppState error:", err.message);
  }

  if (attempt < 3) {
    syncRetryTimer = setTimeout(() => trySyncChats(attempt + 1), 15000);
  }
}

// ── Fetch all participating groups and merge ───────────────────────────────
async function fetchGroups() {
  if (!sock || !clientReady) return 0;
  try {
    const groups = await sock.groupFetchAllParticipating();
    let added = 0;
    for (const [id, meta] of Object.entries(groups)) {
      if (!chatStore[id]) {
        chatStore[id] = {
          id,
          name: meta.subject || id.split("@")[0],
          conversationTimestamp: meta.creation || 0,
          unreadCount: 0,
        };
        added++;
      }
    }
    if (added) console.log(`[WA] groupFetchAllParticipating: ${added} new groups added`);
    return added;
  } catch (err) {
    console.error("[WA] groupFetchAllParticipating error:", err.message);
    return 0;
  }
}

// ── Clear session files ────────────────────────────────────────────────────
function clearAuthFiles() {
  try {
    if (!fs.existsSync(AUTH_DIR)) return;
    const files = fs.readdirSync(AUTH_DIR);
    for (const f of files) fs.unlinkSync(path.join(AUTH_DIR, f));
    console.log(`[WA] Cleared ${files.length} auth file(s) — fresh session required`);
  } catch (err) {
    console.error("[WA] Failed to clear auth files:", err.message);
  }
}

// ── Baileys connection ─────────────────────────────────────────────────────
async function startWhatsApp() {
  if (!fs.existsSync(AUTH_DIR)) fs.mkdirSync(AUTH_DIR, { recursive: true });

  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version }          = await fetchLatestBaileysVersion();
  console.log(`[WA] Baileys version: ${version.join(".")}`);

  sock = makeWASocket({
    version,
    auth: state,
    logger,
    // Ask WhatsApp for history during the initial link. Existing lightweight
    // sessions must pair again before WhatsApp sends historical messages.
    browser: ["RAKSHA AI", "Chrome", "1.0.0"],
    syncFullHistory: true,
    // Suppress SessionError / MessageCounterError for stale messages
    getMessage: async (key) => {
      const msgs = messageStore[key.remoteJid || ""] || [];
      const found = msgs.find((m) => m.id === key.id);
      return found ? { conversation: found.body } : undefined;
    },
  });
  const activeSocket = sock;
  const isActiveSocket = () => sock === activeSocket;

  // ── Connection lifecycle ────────────────────────────────────────────────
  activeSocket.ev.on("connection.update", async (update) => {
    if (!isActiveSocket()) return;
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      try {
        lastQrDataUrl = await QRCode.toDataURL(qr, { width: 300, margin: 2 });
        broadcast("qr", { qr: lastQrDataUrl });
        console.log("[WA] QR ready — scan with your phone");
      } catch {}
    }

    if (connection === "open") {
      clientReady = true;
      lastQrDataUrl = null;
      try {
        const me   = activeSocket.user;
        connectedPhone = me?.id?.split(":")[0] || me?.id?.split("@")[0] || null;
        console.log(`[WA] ✅ Connected! Phone: +${connectedPhone}`);
      } catch { connectedPhone = "unknown"; }

      broadcast("ready", { connected: true, phone: connectedPhone });

      // Give Baileys 3 s to fire chats.set from its own init, then try manual resync
      syncRetryTimer = setTimeout(() => trySyncChats(1), 3000);
      // Log stats after delays regardless
      setTimeout(() => logStats("5s after connect"), 5000);
      setTimeout(() => logStats("15s after connect"), 15000);
    }

    if (connection === "close") {
      const code          = lastDisconnect?.error?.output?.statusCode;
      const shouldReconnect = code !== DisconnectReason.loggedOut;
      clientReady = false;
      if (syncRetryTimer) { clearTimeout(syncRetryTimer); syncRetryTimer = null; }

      console.log(`[WA] Disconnected (code: ${code}). Reconnect: ${shouldReconnect}`);

      if (code === DisconnectReason.loggedOut) {
        connectedPhone = null; lastQrDataUrl = null;
        chatStore = {}; messageStore = {};
        clearAuthFiles();
        broadcast("disconnected", { reason: "logged_out" });
        scheduleReconnect(activeSocket, 2000);
      } else if (shouldReconnect) {
        broadcast("loading", { percent: 0, message: "Reconnecting…" });
        scheduleReconnect(activeSocket, 3000);
      }
    }
  });

  activeSocket.ev.on("creds.update", (...args) => {
    if (isActiveSocket()) saveCreds(...args);
  });

  // ── Chat events ─────────────────────────────────────────────────────────
  activeSocket.ev.on("chats.set", ({ chats = [], isLatest }) => {
    if (!isActiveSocket()) return;
    let added = 0;
    for (const c of chats) { upsertChat(c); added++; }
    logStats(`chats.set (${added} chats, isLatest=${isLatest})`);
  });

  activeSocket.ev.on("chats.upsert",  (chats = [])   => {
    if (isActiveSocket()) chats.forEach(upsertChat);
  });
  activeSocket.ev.on("chats.update",  (updates = [])  => {
    if (isActiveSocket()) updates.forEach(upsertChat);
  });

  activeSocket.ev.on("contacts.upsert", (contacts = []) => {
    if (!isActiveSocket()) return;
    for (const c of contacts) {
      if (!c.id || !isRealChat(c.id)) continue;
      if (!chatStore[c.id]) {
        upsertChat({
          id:   c.id,
          name: c.notify || c.name || c.verifiedName || c.id.split("@")[0],
          conversationTimestamp: 0,
          unreadCount: 0,
        });
      }
    }
  });

  // ── History / message events ────────────────────────────────────────────
  activeSocket.ev.on("messaging-history.set", ({ chats = [], contacts = [], messages = [], isLatest }) => {
    if (!isActiveSocket()) return;
    chats.forEach(upsertChat);

    for (const c of contacts) {
      if (!c.id || !isRealChat(c.id)) continue;
      if (!chatStore[c.id]) {
        upsertChat({
          id: c.id,
          name: c.notify || c.name || c.id.split("@")[0],
          conversationTimestamp: 0,
          unreadCount: 0,
        });
      }
    }

    Promise.all(messages.map(upsertMessage)).then(() => logStats(`messaging-history.set (isLatest=${isLatest})`));
  });

  activeSocket.ev.on("messages.set", ({ messages = [] }) => {
    if (!isActiveSocket()) return;
    Promise.all(messages.map(upsertMessage)).then(() => {
      const total = Object.values(messageStore).reduce((s, a) => s + a.length, 0);
      console.log(`[WA] messages.set: ${total} messages total`);
    });
  });

  activeSocket.ev.on("messages.upsert", async ({ messages = [], type }) => {
    if (!isActiveSocket()) return;
    for (const msg of messages) {
      await upsertMessage(msg);
      if (type === "notify" && msg.message) {
        const chatId = msg.key.remoteJid || "";
        if (isRealChat(chatId)) {
          const saved = (messageStore[chatId] || []).find((m) => m.id === msg.key.id);
          if (saved) {
            broadcast("message", {
              ...saved,
              chatName: chatStore[chatId]?.name || chatId.split("@")[0],
            });
          }
        }
      }
    }
  });
}

// ── REST endpoints ─────────────────────────────────────────────────────────

app.get("/health", (_req, res) => res.json({ status: "ok" }));

app.get("/status", (_req, res) => res.json({
  connected: clientReady,
  qrReady:   !!lastQrDataUrl,
  phone:     connectedPhone,
  historySyncEnabled: true,
}));

app.get("/chats", async (_req, res) => {
  if (!clientReady || !sock)
    return res.status(503).json({ error: "WhatsApp not connected" });

  try {
    // Always refresh group list (direct API call, always works)
    await fetchGroups();

    const all = Object.values(chatStore);

    // Sort by most-recently-active then cap
    const sorted = all
      .sort((a, b) => {
        const ta = typeof a.conversationTimestamp === "number" ? a.conversationTimestamp
          : parseInt(a.conversationTimestamp?.toString() || "0");
        const tb = typeof b.conversationTimestamp === "number" ? b.conversationTimestamp
          : parseInt(b.conversationTimestamp?.toString() || "0");
        return tb - ta;
      })
      .slice(0, MAX_CHATS);

    const result = sorted.map((chat) => ({
      kind:         chatKind(chat.id, chat),
      id:           chat.id,
      name:         chat.name || chat.subject || chat.id?.split("@")[0] || "Unknown",
      isGroup:      isJidGroup(chat.id),
      timestamp:    typeof chat.conversationTimestamp === "number"
        ? chat.conversationTimestamp
        : parseInt(chat.conversationTimestamp?.toString() || "0") || null,
      unreadCount:  chat.unreadCount || 0,
      lastMessage:  lastStoredMessage(chat.id)?.preview || null,
      messageCount: (messageStore[chat.id] || []).length,
      hasMedia:     (messageStore[chat.id] || []).some((m) => m.media && m.media.kind !== "text"),
    }));

    logStats(`/chats → returning ${result.length} of ${all.length}`);
    res.json(result);
  } catch (err) {
    console.error("[API] /chats error:", err.message?.slice(0, 120));
    res.status(500).json({ error: err.message });
  }
});

app.get("/chats/:chatId/messages", (req, res) => {
  if (!clientReady || !sock)
    return res.status(503).json({ error: "WhatsApp not connected" });

  const { chatId } = req.params;
  const limit = Math.min(parseInt(req.query.limit || "40", 10), MAX_MSGS_STORED);
  const messages  = getMessages(chatId, limit);
  const chatMeta  = chatStore[chatId];

  console.log(`[API] /chats/${chatId.slice(0, 20)}/messages → ${messages.length} messages`);

  res.json({
    chatId,
    chatName: chatMeta?.name || chatMeta?.subject || chatId.split("@")[0] || "Unknown",
    isGroup:  isJidGroup(chatId),
    kind:     chatKind(chatId, chatMeta),
    messages,
    totalFetched: messages.length,
  });
});

app.post("/disconnect", async (_req, res) => {
  if (!sock) return res.json({ disconnected: true });
  try { await sock.logout(); } catch {}
  clientReady = false; connectedPhone = null; lastQrDataUrl = null;
  chatStore = {}; messageStore = {};
  broadcast("disconnected", { reason: "user_requested" });
  res.json({ disconnected: true });
});

// Clear the auth session → user will need to re-scan QR
app.post("/clear-session", async (_req, res) => {
  console.log("[WA] 🔄 Clear session requested");
  if (syncRetryTimer) { clearTimeout(syncRetryTimer); syncRetryTimer = null; }
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  if (sock) { try { sock.end(undefined); } catch {} sock = null; }
  clientReady = false; connectedPhone = null; lastQrDataUrl = null;
  chatStore = {}; messageStore = {};

  broadcast("disconnected", { reason: "session_cleared" });

  clearAuthFiles();
  setTimeout(() => startWhatsApp(), 1500);
  res.json({ cleared: true, message: "Session cleared — new QR code will appear shortly" });
});

app.post("/reinit", async (_req, res) => {
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  if (sock) { try { sock.end(undefined); } catch {} }
  clientReady = false; connectedPhone = null; lastQrDataUrl = null;
  chatStore = {}; messageStore = {};
  startWhatsApp();
  res.json({ message: "Re-initializing — watch for QR via WebSocket" });
});

// Debug snapshot
app.get("/debug", (_req, res) => {
  const all = Object.values(chatStore);
  res.json({
    connected:        clientReady,
    phone:            connectedPhone,
    historySyncEnabled: true,
    totalChats:       all.length,
    personal:         all.filter((c) => isJidUser(c.id)).length,
    groups:           all.filter((c) => isJidGroup(c.id)).length,
    chatsWithMessages: Object.keys(messageStore).filter((id) => messageStore[id].length > 0).length,
    totalMessages:    Object.values(messageStore).reduce((s, a) => s + a.length, 0),
    sampleChats:      all.slice(0, 8).map((c) => ({
      id: c.id, name: c.name,
      isGroup: isJidGroup(c.id),
      messages: (messageStore[c.id] || []).length,
    })),
  });
});

// ── Start ──────────────────────────────────────────────────────────────────
async function boot() {
  try {
    const baileys = await import("@whiskeysockets/baileys");
    ({
      default: makeWASocket,
      useMultiFileAuthState,
      DisconnectReason,
      fetchLatestBaileysVersion,
      downloadMediaMessage,
      getContentType,
      isJidGroup,
      isJidUser,
      isJidBroadcast,
      isJidNewsletter,
    } = baileys);
  } catch (error) {
    console.error("[WA] Failed to load Baileys:", error);
    process.exit(1);
  }

  server.listen(PORT, () => {
    console.log(`\n🟢 WhatsApp Bridge running on http://localhost:${PORT}`);
    console.log(`   Status:  GET  http://localhost:${PORT}/status`);
    console.log(`   Chats:   GET  http://localhost:${PORT}/chats`);
    console.log(`   Reset:   POST http://localhost:${PORT}/clear-session`);
    console.log(`   Debug:   GET  http://localhost:${PORT}/debug\n`);
    startWhatsApp();
  });
}

boot();
