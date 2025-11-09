<template>
  <div class="mb-2">
    <button @click="record" class="bg-blue-500 text-white px-3 py-1 rounded">🎤 Nahrát audio</button>
    <audio v-if="audioUrl" :src="audioUrl" controls class="mt-2 w-full"></audio>
  </div>
</template>

<script setup>
import { ref, defineEmits } from 'vue';

const audioUrl = ref(null);
let mediaRecorder, chunks = [];
const emit = defineEmits(['newAudio']);

async function record() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  chunks = [];
  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.onstop = () => {
    const blob = new Blob(chunks, { type: 'audio/wav' });
    audioUrl.value = URL.createObjectURL(blob);
    emit('newAudio', blob);
  };
  mediaRecorder.start();
  setTimeout(() => mediaRecorder.stop(), 5000); // 5s nahrávka
}
</script>
