<script setup lang="ts">
defineProps<{ label: string; value: string; detail?: string; tone?: 'positive' | 'negative' | 'neutral' }>()
</script>

<template>
  <article class="metric-card" :class="tone || 'neutral'">
    <span>{{ label }}</span><strong>{{ value }}</strong><small v-if="detail">{{ detail }}</small>
  </article>
</template>
