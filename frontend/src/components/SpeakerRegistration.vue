<template>
  <div class="p-4 border rounded mb-4 bg-white shadow">
    <h2 class="text-xl font-bold mb-2">Registrace mluvčího</h2>
    <input v-model="name" placeholder="Jméno" class="border p-2 rounded w-full mb-2"/>
    <div class="flex items-center gap-2">
      <button @click="record" class="bg-blue-500 text-white px-3 py-1 rounded">🎙️ Nahrát 3s</button>
      <button v-if="audioBlob" @click="upload" class="bg-green-500 text-white px-3 py-1 rounded">Uložit</button>
    </div>
    <audio v-if="audioUrl" :src="audioUrl" controls class="mt-2 w-full"></audio>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { registerSpeaker } from '../api/backend.js';

const name = ref('');
const audioBlob = ref(null);
const audioUrl = ref(null);
let mediaRecorder, chunks = [];

async function record() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  chunks = [];
  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.onstop = () => {
    audioBlob.value = new Blob(chunks, { type: 'audio/wav' });
    audioUrl.value = URL.createObjectURL(audioBlob.value);
  };
  mediaRecorder.start();
  setTimeout(() => mediaRecorder.stop(), 3000);
}

async function upload() {
  if (!name.value || !audioBlob.value) return alert("Zadej jméno a nahraj audio!");
  const res = await registerSpeaker(name.value, audioBlob.value);
  alert(`Mluvčí ${res.speaker} zaregistrován`);
  name.value = '';
  audioBlob.value = null;
  audioUrl.value = null;
}
</script>
