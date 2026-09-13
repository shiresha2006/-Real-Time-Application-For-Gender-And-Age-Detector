const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function fetchStatus() {
  const res = await fetch(`${BASE_URL}/api/status`);
  return res.json();
}

export async function predictImageFile(file, bypassHours) {
  const form = new FormData();
  form.append("file", file);
  form.append("bypass_hours", bypassHours ? "true" : "false");
  const res = await fetch(`${BASE_URL}/api/predict/image`, {
    method: "POST",
    body: form,
  });
  return res.json();
}

export async function predictFrame(dataUrl, bypassHours) {
  const res = await fetch(`${BASE_URL}/api/predict/frame`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl, bypass_hours: !!bypassHours }),
  });
  return res.json();
}
