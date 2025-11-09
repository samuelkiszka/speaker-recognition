<template>
  <div class="mt-2 max-h-64 overflow-y-auto p-2 bg-gray-50 border rounded">
    <div v-for="(seg, idx) in segments" :key="idx" class="mb-1">
      <span class="font-semibold">{{ formatTime(seg.start) }} {{ seg.speaker }}:</span>
      <span>{{ seg.text }}</span>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  segments: {
    type: Array,
    default: () => []
  }
});

function formatTime(sec) {
  const m = Math.floor(sec / 60).toString().padStart(2, '0');
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `[${m}:${s}]`;
}
</script>
