const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const dzEmpty = document.getElementById('dzEmpty');
const previewImg = document.getElementById('previewImg');
const detectBtn = document.getElementById('detectBtn');
const statusEl = document.getElementById('status');
const resultImg = document.getElementById('resultImg');
const resultEmpty = document.getElementById('resultEmpty');
const carCountEl = document.getElementById('carCount');
const peopleCountEl = document.getElementById('peopleCount');

let selectedFile = null;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.hidden = false;
  dzEmpty.hidden = true;
  detectBtn.disabled = false;
  statusEl.textContent = '';
}

detectBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  detectBtn.disabled = true;
  statusEl.textContent = 'Running detection…';
  const form = new FormData();
  form.append('image', selectedFile);

  try {
    const res = await fetch('/api/detect', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Detection failed');

    resultImg.src = data.image;
    resultImg.hidden = false;
    resultEmpty.hidden = true;
    carCountEl.textContent = data.car_count;
    peopleCountEl.textContent = data.people_count;
    statusEl.textContent = `Done — ${data.car_count} car(s), ${data.people_count} person(es) detected.`;
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  } finally {
    detectBtn.disabled = false;
  }
});
