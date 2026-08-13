import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE_URL || ''

export const api = axios.create({ baseURL, timeout: 120_000 })

api.interceptors.request.use((config) => {
  const token = import.meta.env.VITE_ADMIN_API_TOKEN
  if (token) config.headers['X-Admin-Token'] = token
  return config
})

api.interceptors.response.use(undefined, async (error) => {
  const config = error.config as (typeof error.config & { retryCount?: number }) | undefined
  const retryable = config?.method?.toLowerCase() === 'get' && (!error.response || [502, 503, 504].includes(error.response.status))
  if (!config || !retryable || (config.retryCount || 0) >= 3) return Promise.reject(error)
  config.retryCount = (config.retryCount || 0) + 1
  await new Promise((resolve) => window.setTimeout(resolve, config.retryCount * 5_000))
  return api.request(config)
})

export const errorMessage = (error: unknown) => {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.detail || error.message
  }
  return error instanceof Error ? error.message : String(error)
}
