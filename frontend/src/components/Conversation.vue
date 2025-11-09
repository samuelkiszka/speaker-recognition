<template>
  <div class="p-4 border rounded bg-white shadow">
    <h2 class="text-xl font-bold mb-2">Přepis konverzace</h2>
    <AudioRecorder @newAudio="transcribeAudio"/>
    <TranscriptView :segments="segments"/>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import AudioRecorder from './AudioRecorder.vue';
import TranscriptView from './TranscriptView.vue';
import { transcribeAudioFile } from '../api/backend.js';

const segments = ref([]);

async function transcribeAudio(audioBlob) {
  const res = await transcribeAudioFile(audioBlob);
  segments.value = res.segments;
}
</script>
