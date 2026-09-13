const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";

export async function fetchStatus() {
  const res = await fetch(`${BASE_URL}/api/status`);
  return res.json();
}

export async function detectFrame(dataUrl, source = "live") {
  const res = await fetch(`${BASE_URL}/api/detect/frame`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl, source }),
  });
  return res.json();
}

export async function detectVideo(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/api/detect/video`, { method: "POST", body: form });
  return res.json();
}

export async function fetchLog() {
  const res = await fetch(`${BASE_URL}/api/log`);
  return res.json();
}

export async function clearLog() {
  const res = await fetch(`${BASE_URL}/api/log/clear`, { method: "POST" });
  return res.json();
}

export function csvDownloadUrl() {
  return `${BASE_URL}/api/log/csv`;
}

export function excelDownloadUrl() {
  return `${BASE_URL}/api/log/excel`;
}
