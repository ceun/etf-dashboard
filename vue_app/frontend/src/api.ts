import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE_URL || ''

export const api = axios.create({ baseURL, timeout: 90_000 })

api.interceptors.request.use((config) => {
  const token = import.meta.env.VITE_ADMIN_API_TOKEN
  if (token) config.headers['X-Admin-Token'] = token
  return config
})

export const errorMessage = (error: unknown) => {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.detail || error.message
  }
  return error instanceof Error ? error.message : String(error)
}
