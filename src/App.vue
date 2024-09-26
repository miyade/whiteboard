<!-- src/views/HomeView.vue -->
<template>
  <div class="home">
    <h1>AI Whiteboard</h1>
    <Whiteboard />
  </div>
</template>

<script>
import Whiteboard from '@/components/WhiteBoard.vue';

export default {
  name: 'HomeView',
  components: {
    Whiteboard,
  },
};
</script>

<style scoped>
</style>
