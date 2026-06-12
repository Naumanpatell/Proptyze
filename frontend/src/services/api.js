import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
})

export const uploadFiles = (files, onProgress) => {
  const form = new FormData()
  files.forEach(file => form.append('files', file))
  return api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => onProgress?.(Math.round((e.loaded * 100) / e.total)),
  })
}

export const getScanStatus = (scanId) => api.get(`/scans/${scanId}/status`)
export const getReports = () => api.get('/reports')
export const getReport = (id) => api.get(`/reports/${id}`)
export const deleteReport = (id) => api.delete(`/reports/${id}`)
export const downloadReportPdf = (id) => api.get(`/reports/${id}/pdf`, { responseType: 'blob' })
export const getNeighbourhood = (postcode) => api.get(`/neighbourhood/${postcode}`)
export const getHealth = () => api.get('/health')

export default api
