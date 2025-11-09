const BASE_URL = "http://localhost:8000";

export async function registerSpeaker(name, blob) {
  const form = new FormData();
  form.append("name", name);
  form.append("file", blob, "audio.wav");
  const res = await fetch(`${BASE_URL}/register`, { method: "POST", body: form });
  return await res.json();
}

export async function transcribeAudioFile(blob) {
  const form = new FormData();
  form.append("file", blob, "audio.wav");
  const res = await fetch(`${BASE_URL}/transcribe`, { method: "POST", body: form });
  return await res.json();
}
