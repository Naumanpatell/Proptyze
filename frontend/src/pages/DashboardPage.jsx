import { useEffect, useState, useCallback } from 'react'
import { useSearchParams, useNavigate, Link } from 'react-router-dom'
import { getScanStatus, getReports, deleteReport } from '../services/api'

const STAGE_LABELS = {
  waiting:           'Waiting to start',
  extracting_frames: 'Extracting video frames',
  frames_ready:      'Frames extracted',
  detecting:         'Detecting issues',
  scoring:           'Computing property score',
  generating_report: 'Generating AI report',
  done:              'Complete',
  error:             'Analysis failed',
}

const GRADE_COLORS = {
  A: 'bg-green-100 text-green-700 ring-green-300',
  B: 'bg-lime-100 text-lime-700 ring-lime-300',
  C: 'bg-yellow-100 text-yellow-700 ring-yellow-300',
  D: 'bg-red-100 text-red-700 ring-red-300',
}

function ScoreRing({ score }) {
  const radius = 36
  const circ = 2 * Math.PI * radius
  const offset = circ * (1 - score / 100)
  const color = score >= 80 ? '#22c55e' : score >= 60 ? '#84cc16' : score >= 40 ? '#eab308' : '#ef4444'
  return (
    <svg width="96" height="96" viewBox="0 0 96 96" className="rotate-[-90deg]">
      <circle cx="48" cy="48" r={radius} fill="none" stroke="#e5e7eb" strokeWidth="8" />
      <circle
        cx="48" cy="48" r={radius}
        fill="none" stroke={color} strokeWidth="8"
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.8s ease' }}
      />
    </svg>
  )
}

function ReportCard({ report, onDelete }) {
  const [confirming, setConfirming] = useState(false)

  const handleDelete = async () => {
    if (!confirming) { setConfirming(true); return }
    await deleteReport(report.id)
    onDelete(report.id)
  }

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-5 flex items-center gap-5 shadow-sm hover:shadow-md transition-shadow">
      {/* Score ring */}
      <div className="relative flex-shrink-0 w-24 h-24 flex items-center justify-center">
        <ScoreRing score={report.score} />
        <span className="absolute text-lg font-bold text-gray-800">{report.score}</span>
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ring-1 ${GRADE_COLORS[report.grade] || GRADE_COLORS.D}`}>
            Grade {report.grade}
          </span>
        </div>
        <p className="text-sm font-medium text-gray-800 truncate">{report.filename || 'Unknown file'}</p>
        <p className="text-xs text-gray-400 mt-0.5">
          {new Date(report.created_at).toLocaleDateString('en-GB', {
            day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
          })}
        </p>
      </div>

      <div className="flex items-center gap-2 flex-shrink-0">
        <Link
          to={`/reports/${report.id}`}
          className="px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 transition-colors"
        >
          View
        </Link>
        <button
          onClick={handleDelete}
          className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
            confirming
              ? 'bg-red-600 text-white hover:bg-red-700'
              : 'border border-gray-200 text-gray-500 hover:text-red-500 hover:border-red-300'
          }`}
        >
          {confirming ? 'Confirm?' : 'Delete'}
        </button>
      </div>
    </div>
  )
}

function ScanTracker({ scanId, onComplete }) {
  const [scan, setScan] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    let timer

    const poll = async () => {
      try {
        const { data } = await getScanStatus(scanId)
        if (cancelled) return
        setScan(data)
        if (data.status === 'complete') {
          onComplete(data.report_id)
          return
        }
        if (data.status === 'failed') return
        timer = setTimeout(poll, 2000)
      } catch (err) {
        if (!cancelled) setError('Could not reach the server. Retrying…')
        timer = setTimeout(poll, 4000)
      }
    }

    poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [scanId, onComplete])

  const stage = scan?.stage || 'waiting'
  const progress = scan?.progress ?? 0
  const status = scan?.status || 'queued'

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm space-y-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-semibold text-gray-700">
            {status === 'failed' ? 'Analysis failed' : 'Analysing your property…'}
          </p>
          <p className="text-xs text-gray-400 mt-0.5 truncate max-w-xs">{scan?.filename || scanId}</p>
        </div>
        <span className={`text-xs font-medium px-2 py-1 rounded-full ${
          status === 'complete' ? 'bg-green-100 text-green-700' :
          status === 'failed'   ? 'bg-red-100 text-red-700' :
                                  'bg-indigo-100 text-indigo-700'
        }`}>
          {status}
        </span>
      </div>

      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-xs text-gray-400 mb-1.5">
          <span>{STAGE_LABELS[stage] || stage}</span>
          <span>{progress}%</span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
          <div
            className={`h-2.5 rounded-full transition-all duration-700 ${
              status === 'failed' ? 'bg-red-500' : 'bg-indigo-500'
            }`}
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Stage steps */}
      {status !== 'failed' && (
        <div className="grid grid-cols-3 gap-2 pt-1">
          {[
            { key: 'extract',  label: 'Extract frames',  done: progress >= 25 },
            { key: 'detect',   label: 'Detect issues',   done: progress >= 65 },
            { key: 'report',   label: 'AI report',       done: progress >= 100 },
          ].map(({ key, label, done }) => (
            <div key={key} className={`flex items-center gap-1.5 text-xs ${done ? 'text-green-600' : 'text-gray-400'}`}>
              {done ? (
                <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <span className="w-3.5 h-3.5 rounded-full border border-gray-300 flex-shrink-0" />
              )}
              {label}
            </div>
          ))}
        </div>
      )}

      {status === 'failed' && (
        <p className="text-sm text-red-500">
          Something went wrong during analysis. Try uploading again.
        </p>
      )}

      {error && <p className="text-xs text-amber-600">{error}</p>}
    </div>
  )
}

export default function DashboardPage() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const scanId = searchParams.get('scan_id')

  const [reports, setReports] = useState([])
  const [loading, setLoading] = useState(true)

  const loadReports = useCallback(async () => {
    try {
      const { data } = await getReports()
      setReports(data.reports)
    } catch {
      // silently fail — list just won't show
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadReports() }, [loadReports])

  const handleScanComplete = useCallback((reportId) => {
    if (reportId) {
      setTimeout(() => navigate(`/reports/${reportId}`), 2500)
    }
    loadReports()
  }, [navigate, loadReports])

  const handleDelete = useCallback((deletedId) => {
    setReports(prev => prev.filter(r => r.id !== deletedId))
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-12">
      <div className="max-w-2xl mx-auto space-y-8">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
            <p className="text-sm text-gray-500 mt-0.5">Your property analysis reports</p>
          </div>
          <Link
            to="/upload"
            className="px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 transition-colors"
          >
            + New analysis
          </Link>
        </div>

        {/* Live scan tracker */}
        {scanId && (
          <section>
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">In progress</h2>
            <ScanTracker scanId={scanId} onComplete={handleScanComplete} />
          </section>
        )}

        {/* Reports list */}
        <section>
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Past reports</h2>

          {loading ? (
            <div className="space-y-3">
              {[0, 1, 2].map(i => (
                <div key={i} className="bg-white rounded-2xl border border-gray-100 p-5 h-24 animate-pulse" />
              ))}
            </div>
          ) : reports.length === 0 ? (
            <div className="bg-white rounded-2xl border border-dashed border-gray-200 p-10 text-center">
              <p className="text-gray-400 text-sm">No reports yet.</p>
              <Link to="/upload" className="text-indigo-600 text-sm font-medium mt-1 inline-block hover:underline">
                Upload your first video →
              </Link>
            </div>
          ) : (
            <div className="space-y-3">
              {reports.map(r => (
                <ReportCard key={r.id} report={r} onDelete={handleDelete} />
              ))}
            </div>
          )}
        </section>

      </div>
    </div>
  )
}
