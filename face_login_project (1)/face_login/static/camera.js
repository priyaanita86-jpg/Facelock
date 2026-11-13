async function startCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia(
    { video: { width: 640, height: 480, facingMode: "user" }, audio: false });
  video.srcObject = stream;
  await video.play();
  return stream;
}

function stopCamera(stream) {
  stream.getTracks().forEach(t => t.stop());
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function captureFrames(video, count, intervalMs, onProgress) {
  const canvas = document.createElement("canvas");
  canvas.width = 480;
  canvas.height = Math.round(480 * video.videoHeight / video.videoWidth);
  const ctx = canvas.getContext("2d");
  const frames = [];
  for (let i = 0; i < count; i++) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    frames.push(canvas.toDataURL("image/jpeg", 0.8));
    if (onProgress) onProgress(i + 1, count);
    await sleep(intervalMs);
  }
  return frames;
}

async function postJSON(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
}
