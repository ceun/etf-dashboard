<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { Database, FileUp, RefreshCw, Trash2 } from 'lucide-vue-next'
import { api, errorMessage } from '../api'
import { appState, loadTargets } from '../state'

interface SummaryRow { index_code: string; name: string; etf_code: string; data_source: string; asset_currency: string; row_count: number; first_date: string; latest_date: string }
const summary = ref<SummaryRow[]>([])
const busy = ref(false)
const message = ref('')
const importForm = reactive({ name: '', index_code: '', etf_code: '', data_source: 'YHE', asset_currency: 'USD', file: null as File | null })
const directForm = reactive({ name: '', index_code: '', etf_code: '', data_source: 'YH', asset_currency: 'USD' })

async function loadSummary() {
  try { summary.value = (await api.get('/api/database/summary')).data } catch (error) { message.value = errorMessage(error) }
}
onMounted(loadSummary)

async function importTarget() {
  if (!importForm.file) { message.value = '请选择历史文件'; return }
  const selectedFile = importForm.file
  busy.value = true; message.value = ''
  const form = new FormData()
  form.append('name', importForm.name)
  form.append('index_code', importForm.index_code)
  form.append('etf_code', importForm.etf_code)
  form.append('data_source', importForm.data_source)
  form.append('asset_currency', importForm.asset_currency)
  form.append('file', selectedFile)
  try {
    const { data } = await api.post('/api/targets/import', form)
    message.value = `导入成功：解析 ${data.parsed_rows} 条，写入 ${data.written_rows} 条`
    await Promise.all([loadTargets(), loadSummary()])
  } catch (error) { message.value = errorMessage(error) } finally { busy.value = false }
}

async function createDirectTarget() {
  busy.value = true; message.value = ''
  try {
    const { data: target } = await api.post('/api/targets', directForm)
    const { data: synced } = await api.post(`/api/targets/${encodeURIComponent(target.index_code)}/sync`)
    message.value = `新增并同步成功：写入 ${synced.written_rows} 条`
    await Promise.all([loadTargets(), loadSummary()])
  } catch (error) { message.value = errorMessage(error) } finally { busy.value = false }
}

async function syncTarget(indexCode: string) {
  busy.value = true; message.value = ''
  try { const { data } = await api.post(`/api/targets/${encodeURIComponent(indexCode)}/sync`); message.value = `${indexCode}: ${data.status}，写入 ${data.written_rows} 条`; await loadSummary() }
  catch (error) { message.value = errorMessage(error) } finally { busy.value = false }
}

async function deleteTarget(indexCode: string, name: string) {
  if (!confirm(`确认删除 ${name} 及其全部价格数据？此操作不可撤销。`)) return
  busy.value = true
  try { const { data } = await api.delete(`/api/targets/${encodeURIComponent(indexCode)}`); message.value = `已删除 ${data.deleted_prices} 条价格记录`; await Promise.all([loadTargets(), loadSummary()]) }
  catch (error) { message.value = errorMessage(error) } finally { busy.value = false }
}
</script>

<template>
  <section class="page-section">
    <div class="section-heading"><div><span class="eyebrow">DATA PIPELINES</span><h2>数据管理</h2><p>导入官方历史，再由 ETF 延伸更新</p></div></div>
    <div v-if="message" class="info-banner">{{ message }}</div>
    <div class="two-column">
      <article class="panel form-panel">
        <div class="panel-heading"><div><h3><FileUp :size="18" /> 新增拼接标的</h3><p>适用于 ZZ 或 YHE 链路</p></div></div>
        <div class="form-grid">
          <label>标的名称<input v-model="importForm.name" placeholder="纳斯达克100全收益" /></label>
          <label>指数代码<input v-model="importForm.index_code" placeholder="^XNDX" /></label>
          <label>关联 ETF<input v-model="importForm.etf_code" placeholder="QQQ" /></label>
          <label>数据源<select v-model="importForm.data_source"><option>YHE</option><option>ZZ</option></select></label>
          <label>资产币种<select v-model="importForm.asset_currency"><option>CNY</option><option>USD</option><option>HKD</option><option>JPY</option><option>EUR</option></select></label>
          <label class="file-label">历史文件<input type="file" accept=".csv,.xlsx,.xls" @change="importForm.file = ($event.target as HTMLInputElement).files?.[0] || null" /><span>{{ importForm.file?.name || '选择 CSV / Excel' }}</span></label>
        </div>
        <button class="primary-button" :disabled="busy" @click="importTarget"><Database :size="16" />{{ busy ? '处理中…' : '解析、拼接并入库' }}</button>
      </article>
      <article class="panel guide-panel"><span class="eyebrow">字段口径</span><h3>最终分析使用什么？</h3><div class="formula">有指数：index_close<br />无指数：ETF 后复权 × 缩放系数<br /><b>combined_close = 原币价格 × 汇率</b></div><p>指数历史永远优先。ETF 只补齐指数缺失日期，不覆盖已有官方历史。</p></article>
    </div>
    <article class="panel form-panel">
      <div class="panel-heading"><div><h3><Database :size="18" /> 新增在线直连标的</h3><p>YH 使用 Yahoo 指数，SZ 使用深证/国证接口；创建后立即首次同步</p></div></div>
      <div class="inline-form">
        <label>标的名称<input v-model="directForm.name" placeholder="纳斯达克100" /></label>
        <label>指数代码<input v-model="directForm.index_code" placeholder="^NDX" /></label>
        <label>关联 ETF（可选）<input v-model="directForm.etf_code" placeholder="QQQ" /></label>
        <label>数据源<select v-model="directForm.data_source"><option>YH</option><option>SZ</option></select></label>
        <label>资产币种<select v-model="directForm.asset_currency"><option>CNY</option><option>USD</option><option>HKD</option><option>JPY</option><option>EUR</option></select></label>
        <button class="primary-button" :disabled="busy || !directForm.name || !directForm.index_code" @click="createDirectTarget">新增并同步</button>
      </div>
    </article>
    <article class="panel table-panel">
      <div class="panel-heading"><div><h3>数据库概览</h3><p>{{ summary.length }} 个已入库标的</p></div></div>
      <div class="table-scroll"><table><thead><tr><th>标的</th><th>链路</th><th>币种</th><th>记录数</th><th>日期范围</th><th>操作</th></tr></thead><tbody>
        <tr v-for="row in summary" :key="row.index_code"><td><strong>{{ row.name }}</strong><small>{{ row.index_code }} · {{ row.etf_code || '未绑定 ETF' }}</small></td><td><span class="source-badge">{{ row.data_source }}</span></td><td>{{ row.asset_currency }}</td><td>{{ row.row_count.toLocaleString() }}</td><td>{{ row.first_date }} → {{ row.latest_date }}</td><td><div class="row-actions"><button class="subtle-button" :disabled="busy" @click="syncTarget(row.index_code)"><RefreshCw :size="14" />同步</button><button class="danger-button" :disabled="busy" @click="deleteTarget(row.index_code, row.name)"><Trash2 :size="14" />删除</button></div></td></tr>
      </tbody></table></div>
    </article>
  </section>
</template>
