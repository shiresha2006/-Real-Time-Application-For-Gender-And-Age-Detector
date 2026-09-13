import { useEffect, useState } from "react";
import "./App.css";
import Sidebar from "./components/Sidebar.jsx";
import UploadStage from "./components/UploadStage.jsx";
import LiveStage from "./components/LiveStage.jsx";
import { fetchStatus } from "./api.js";

export default function App() {
  const [status, setStatus] = useState(null);
  const [activeTab, setActiveTab] = useState("upload");
  const [bypassHours, setBypassHours] = useState(false);
  const [lastMatchedWord, setLastMatchedWord] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const s = await fetchStatus();
        if (!cancelled) setStatus(s);
      } catch {
        if (!cancelled) setStatus(null);
      }
    }
    poll();
    const id = setInterval(poll, 30000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="shell">
      <Sidebar
        status={status}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        bypassHours={bypassHours}
        onBypassChange={setBypassHours}
        lastMatchedWord={lastMatchedWord}
      />
      {activeTab === "upload" ? (
        <UploadStage bypassHours={bypassHours} onMatched={setLastMatchedWord} />
      ) : (
        <LiveStage bypassHours={bypassHours} onMatched={setLastMatchedWord} />
      )}
    </div>
  );
}
