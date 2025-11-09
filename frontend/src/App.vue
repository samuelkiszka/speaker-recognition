<template>
  <div class="min-h-screen bg-gray-50 text-gray-800 p-6">
    <h1 class="text-2xl font-bold mb-4">Voice Transformer App</h1>
    <SpeakerRegistration />
    <Conversation />
  </div>
</template>

<script setup>
import SpeakerRegistration from './components/SpeakerRegistration.vue'
import Conversation from './components/Conversation.vue'
</script>
