const WORDS = ["Hello", "Yes", "Peace", "Good", "Stop", "ILoveYou", "OK", "Point"];

export default function Sidebar({
  status,
  activeTab,
  onTabChange,
  bypassHours,
  onBypassChange,
  lastMatchedWord,
}) {
  const isOpen = status?.is_open;

  return (
    <aside className="rail">
      <div className="brand">
        <div className="brand-mark">
          <span className="dot" />
          Signal
        </div>
        <p className="brand-sub">
          Sign language detection terminal. Reads a hand pose from an image
          or live camera and matches it against a small glossary of known
          words.
        </p>
      </div>

      <div className="sign-badge">
        <div className={`sign-state ${isOpen ? "open" : "closed"}`}>
          <span className="bulb" />
          {status ? (isOpen ? "Open" : "Closed") : "Connecting…"}
        </div>
        <div className="sign-hours">
          {status ? `Serves detections ${status.open_time}–${status.close_time}` : "—"}
        </div>
        <div className="sign-clock">
          {status ? `Server time ${status.server_time}` : "—"}
        </div>
      </div>

      <label className="bypass-row">
        <input
          type="checkbox"
          checked={bypassHours}
          onChange={(e) => onBypassChange(e.target.checked)}
        />
        Demo mode — ignore operating hours
      </label>

      <nav className="nav">
        <div className="nav-label">Input</div>
        <button
          className={`nav-item ${activeTab === "upload" ? "active" : ""}`}
          onClick={() => onTabChange("upload")}
        >
          Upload image
        </button>
        <button
          className={`nav-item ${activeTab === "live" ? "active" : ""}`}
          onClick={() => onTabChange("live")}
        >
          Live camera
        </button>
      </nav>

      <div className="glossary">
        <div className="nav-label">Known words</div>
        <div className="glossary-chips">
          {WORDS.map((w) => (
            <span key={w} className={`chip ${w === lastMatchedWord ? "matched" : ""}`}>
              {w}
            </span>
          ))}
        </div>
      </div>
    </aside>
  );
}
