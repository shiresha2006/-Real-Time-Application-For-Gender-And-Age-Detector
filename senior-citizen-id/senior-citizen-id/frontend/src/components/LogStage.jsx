import { useEffect, useState } from "react";
import { fetchLog, clearLog, csvDownloadUrl, excelDownloadUrl } from "../api.js";

export default function LogStage({ refreshKey, onCleared }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      const res = await fetchLog();
      setRows(res.rows || []);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, [refreshKey]);

  async function handleClear() {
    if (!window.confirm("Clear the entire visitor log? This can't be undone.")) return;
    await clearLog();
    onCleared();
    load();
  }

  return (
    <div className="stage">
      <div className="stage-header">
        <h1>Visitor log</h1>
        <p>Every logged visit — one row per detected person, deduplicated across frames.</p>
      </div>

      <div className="controls-row">
        <a className="btn secondary" href={csvDownloadUrl()} download>Download CSV</a>
        <a className="btn secondary" href={excelDownloadUrl()} download>Download Excel</a>
        <button className="btn danger" onClick={handleClear}>Clear log</button>
      </div>

      <div className="log-table-wrap">
        {loading ? (
          <div className="log-empty">Loading…</div>
        ) : rows.length === 0 ? (
          <div className="log-empty">No visits logged yet. Run live camera or process a video to populate this table.</div>
        ) : (
          <table className="log-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Gender</th>
                <th>Est. age</th>
                <th>Senior citizen</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {[...rows].reverse().map((r, i) => (
                <tr key={i}>
                  <td className="mono">{r.timestamp}</td>
                  <td>{r.gender}</td>
                  <td className="mono">{r.estimated_age}</td>
                  <td>{r.senior_citizen}</td>
                  <td>{r.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
