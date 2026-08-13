<script setup lang="ts">
import { onMounted } from 'vue'
import { RouterView } from 'vue-router'
import Sidebar from './components/Sidebar.vue'
import TopBar from './components/TopBar.vue'
import { appState, loadTargets } from './state'

onMounted(loadTargets)
</script>

<template>
  <div class="app-shell">
    <Sidebar />
    <div v-if="appState.sidebarOpen" class="mobile-backdrop" @click="appState.sidebarOpen = false" />
    <main class="main-content">
      <TopBar />
      <RouterView />
    </main>
  </div>
</template>
