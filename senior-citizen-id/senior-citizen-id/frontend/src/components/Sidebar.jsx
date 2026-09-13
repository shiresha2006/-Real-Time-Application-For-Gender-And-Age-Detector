export default function Sidebar({ activeTab, onTabChange, totalVisits, seniorCount }) {
  return (
    <aside className="rail">
      <div>
        <div className="brand-mark">
          <span className="dot" />
          Foot Traffic
        </div>
        <p className="brand-sub">
          Detects visitors from a store or mall camera, estimates age and
          gender, and flags anyone over 60 so staff can offer priority
          assistance.
        </p>
      </div>

      <div className="stat-row">
        <div className="stat-tile">
          <div className="stat-number">{totalVisits}</div>
          <div className="stat-label">Visits logged</div>
        </div>
        <div className="stat-tile gold">
          <div className="stat-number">{seniorCount}</div>
          <div className="stat-label">Senior citizens</div>
        </div>
      </div>

      <nav className="nav">
        <div className="nav-label">Input</div>
        <button className={`nav-item ${activeTab === "live" ? "active" : ""}`} onClick={() => onTabChange("live")}>
          Live camera
        </button>
        <button className={`nav-item ${activeTab === "video" ? "active" : ""}`} onClick={() => onTabChange("video")}>
          Upload video
        </button>
        <button className={`nav-item ${activeTab === "log" ? "active" : ""}`} onClick={() => onTabChange("log")}>
          Visitor log
        </button>
      </nav>

      <p className="privacy-note">
        Runs locally against the frames you provide. No face images are
        stored — only the estimated age, gender, and timestamp go into the
        log. Age and gender are model estimates and may be inaccurate for
        any individual.
      </p>
    </aside>
  );
}
