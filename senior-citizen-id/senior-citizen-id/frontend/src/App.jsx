import { useCallback, useEffect, useState } from "react";
import "./App.css";
import Sidebar from "./components/Sidebar.jsx";
import LiveStage from "./components/LiveStage.jsx";
import VideoStage from "./components/VideoStage.jsx";
import LogStage from "./components/LogStage.jsx";
import { fetchLog } from "./api.js";

export default function App() {
  const [activeTab, setActiveTab] = useState("live");
  const [refreshKey, setRefreshKey] = useState(0);
  const [summary, setSummary] = useState({ total: 0, seniors: 0 });

  const refreshSummary = useCallback(async () => {
    try {
      const res = await fetchLog();
      const rows = res.rows || [];
      setSummary({
        total: rows.length,
        seniors: rows.filter((r) => r.senior_citizen === "Yes").length,
      });
    } catch {
      // leave previous summary in place if the backend is briefly unreachable
    }
  }, []);

  useEffect(() => {
    refreshSummary();
  }, [refreshKey, refreshSummary]);

  function bumpRefresh() {
    setRefreshKey((k) => k + 1);
  }

  return (
    <div className="shell">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        totalVisits={summary.total}
        seniorCount={summary.seniors}
      />
      {activeTab === "live" && <LiveStage onLogged={bumpRefresh} />}
      {activeTab === "video" && <VideoStage onLogged={bumpRefresh} />}
      {activeTab === "log" && <LogStage refreshKey={refreshKey} onCleared={bumpRefresh} />}
    </div>
  );
}
