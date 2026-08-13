import { createRouter, createWebHistory } from 'vue-router'
import DetailView from './views/DetailView.vue'
import ComparisonView from './views/ComparisonView.vue'
import DataManagementView from './views/DataManagementView.vue'
import RotationView from './views/RotationView.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'detail', component: DetailView },
    { path: '/comparison', name: 'comparison', component: ComparisonView },
    { path: '/data', name: 'data', component: DataManagementView },
    { path: '/rotation', name: 'rotation', component: RotationView },
  ],
})
