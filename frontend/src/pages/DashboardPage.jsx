import { useEffect, useState, useCallback, useRef } from 'react'
import { useSearchParams, useNavigate, Link } from 'react-router-dom'
import { getScanStatus, getReports, deleteReport } from '../services/api'

const STAGE_LABELS = {
  waiting:           'Waiting to start',
  extracting_frames: 'Extracting video frames',
  frames_ready:      'Frames extracted',
  detecting:         'Running AI detection',
  scoring:           'Computing property score',
  generating_report: 'Generating AI report',
  done:              'Complete',
  error:             'Analysis failed',
}

const GRADE_COLORS = {
  A: 'bg-green-100 text-green-700 ring-green-300',
  B: 'bg-lime-100  text-lime-700  ring-lime-300',
  C: 'bg-yellow-100 text-yellow-700 ring-yellow-300',
  D: 'bg-red-100   text-red-700   ring-red-300',
}

const SCORE_COLOR = (s) =>
  s >= 80 ? '#22c55e' : s >= 60 ? '#84cc16' : s >= 40 ? '#eab308' : '#ef4444'

/* ── Score ring (mini, for report cards) ─────────────────────── */
function MiniRing({ score }) {
  const r = 22
  const circ = 2 * Math.PI * r
  const offset = circ * (1 - score / 100)
  return (
    <svg width="60" height="60" viewBox="0 0 60 60" className="rotate-[-90deg]">
      <circle cx="30" cy="30" r={r} fill="none" stroke="#e5e7eb" strokeWidth="5" />
      <circle
        cx="30" cy="30" r={r}
        fill="none" stroke={SCORE_COLOR(score)} strokeWidth="5"
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(0.16,1,0.3,1)' }}
      />
    </svg>
  )
}

/* ── Single report card ──────────────────────────────────────── */
function ReportCard({ report, index, onDelete }) {
  const [confirming, setConfirming] = useState(false)
  const timerRef = useRef(null)

  // Auto-cancel confirm after 3 s
  useEffect(() => {
    if (!confirming) return
    timerRef.current = setTimeout(() => setConfirming(false), 3000)
    return () => clearTimeout(timerRef.current)
  }, [confirming])

  const handleDelete = async () => {
    if (!confirming) { setConfirming(true); return }
    try {
      await deleteReport(report.id)
      onDelete(report.id)
    } catch {
      setConfirming(false)
    }
  }

  return (
    <div
      className="animate-slide-up group bg-white rounded-2xl border border-gray-200 p-4 flex items-center gap-4
                 shadow-sm hover:shadow-lg hover:-translate-y-0.5 transition-all duration-200"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      {/* Score ring */}
      <div className="relative flex-shrink-0 w-[60px] h-[60px] flex items-center justify-center">
        <MiniRing score={report.score} />
        <span className="absolute text-sm font-bold text-gray-800 animate-score-count"
              style={{ animationDelay: `${index * 60 + 200}ms` }}>
          {report.score}
        </span>
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className={`animate-pop-in text-xs font-semibold px-2 py-0.5 rounded-full ring-1 ${GRADE_COLORS[report.grade] || GRADE_COLORS.D}`}
                style={{ animationDelay: `${index * 60 + 150}ms` }}>
            Grade {report.grade}
          </span>
        </div>
        <p className="text-sm font-medium text-gray-800 truncate">{report.filename || 'Unknown file'}</p>
        <p className="text-xs text-gray-400 mt-0.5">
          {new Date(report.created_at).toLocaleDateString('en-GB', {
            day: 'numeric', month: 'short', year: 'numeric',
            hour: '2-digit', minute: '2-digit',
          })}
        </p>
      </div>

      <div className="flex items-center gap-2 flex-shrink-0">
        <Link
          to={`/reports/${report.id}`}
          className="px-3 py-1.5 rounded-lg bg-indigo-600 text-white text-xs font-medium
                     hover:bg-indigo-700 active:scale-95 transition-all"
        >
          View
        </Link>
        <button
          onClick={handleDelete}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all active:scale-95 ${
            confirming
              ? 'bg-red-500 text-white hover:bg-red-600 animate-pop-in'
              : 'border border-gray-200 text-gray-400 hover:text-red-500 hover:border-red-300'
          }`}
        >
          {confirming ? 'Confirm?' : 'Delete'}
        </button>
      </div>
    </div>
  )
}

/* ── Skeleton placeholder ────────────────────────────────────── */
function Skeleton() {
  return (
    <div className="space-y-3">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="bg-white rounded-2xl border border-gray-100 h-[72px] animate-pulse"
          style={{ animationDelay: `${i * 80}ms` }}
        />
      ))}
    </div>
  )
}

/* ── Live scan tracker ───────────────────────────────────────── */
function ScanTracker({ scanId, onComplete }) {
  const [scan, setScan] = useState(null)
  const [serverError, setServerError] = useState(false)

  useEffect(() => {
    let cancelled = false
    let timer

    const poll = async () => {
      try {
        const { data } = await getScanStatus(scanId)
        if (cancelled) return
        setServerError(false)
        setScan(data)
        if (data.status === 'complete') { onComplete(data.report_id); return }
        if (data.status === 'failed')   return
        timer = setTimeout(poll, 2000)
      } catch {
        if (!cancelled) setServerError(true)
        timer = setTimeout(poll, 4000)
      }
    }

    poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [scanId, onComplete])

  const stage    = scan?.stage    ?? 'waiting'
  const progress = scan?.progress ?? 0
  const status   = scan?.status   ?? 'queued'
  const failed   = status === 'failed'
  const complete = status === 'complete'

  const steps = [
    { label: 'Extract frames', done: progress >= 25 },
    { label: 'Detect issues',  done: progress >= 65 },
    { label: 'AI report',      done: progress >= 100 },
  ]

  return (
    <div className="animate-slide-up bg-white rounded-2xl border border-gray-200 shadow-sm p-6 space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            {!failed && !complete && <span className="live-dot" />}
            <p className="text-sm font-semibold text-gray-800">
              {failed ? 'Analysis failed' : complete ? 'Analysis complete!' : 'Analysing your property…'}
            </p>
          </div>
          <p className="text-xs text-gray-400 truncate">{scan?.filename || scanId}</p>
        </div>
        <span className={`flex-shrink-0 text-xs font-semibold px-2.5 py-1 rounded-full animate-pop-in ${
          complete ? 'bg-green-100 text-green-700' :
          failed   ? 'bg-red-100   text-red-700'   :
                     'bg-indigo-100 text-indigo-700'
        }`}>
          {status}
        </span>
      </div>

      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-xs text-gray-400 mb-2">
          <span className="font-medium">{STAGE_LABELS[stage] || stage}</span>
          <span className="font-semibold tabular-nums" style={{ color: failed ? '#ef4444' : '#6366f1' }}>
            {progress}%
          </span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
          <div
            className={`h-2.5 rounded-full transition-all duration-700 ease-out ${
              failed   ? 'bg-red-400' :
              complete ? 'bg-green-500' :
                         'bg-indigo-500 progress-shimmer'
            }`}
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Step indicators */}
      {!failed && (
        <div className="grid grid-cols-3 gap-2">
          {steps.map(({ label, done }, i) => (
            <div
              key={label}
              className={`flex items-center gap-1.5 text-xs transition-colors duration-300 ${
                done ? 'text-green-600' : 'text-gray-400'
              }`}
            >
              <span className={`flex-shrink-0 w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all duration-300 ${
                done ? 'bg-green-500 border-green-500' : 'border-gray-300'
              }`}>
                {done && (
                  <svg className="w-2.5 h-2.5 text-white animate-pop-in" fill="none" viewBox="0 0 24 24"
                       stroke="currentColor" strokeWidth={3}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </span>
              <span>{label}</span>
            </div>
          ))}
        </div>
      )}

      {failed && (
        <p className="text-sm text-red-500 animate-fade-in">
          Something went wrong during analysis. Try uploading a new video.
        </p>
      )}

      {complete && (
        <p className="text-sm text-green-600 animate-fade-in font-medium">
          Redirecting to your report…
        </p>
      )}

      {serverError && (
        <p className="text-xs text-amber-500 animate-fade-in">
          Can't reach the server — retrying…
        </p>
      )}
    </div>
  )
}

/* ── Page ─────────────────────────────────────────────────────── */
export default function DashboardPage() {
  const [searchParams] = useSearchParams()
  const navigate       = useNavigate()
  const scanId         = searchParams.get('scan_id')

  const [reports, setReports] = useState([])
  const [loading, setLoading] = useState(true)

  const loadReports = useCallback(async () => {
    try {
      const { data } = await getReports()
      setReports(data.reports)
    } catch {
      // list stays empty
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadReports() }, [loadReports])

  const handleScanComplete = useCallback((reportId) => {
    loadReports()
    if (reportId) setTimeout(() => navigate(`/reports/${reportId}`), 2500)
  }, [navigate, loadReports])

  const handleDelete = useCallback((deletedId) => {
    setReports(prev => prev.filter(r => r.id !== deletedId))
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-12">
      <div className="max-w-2xl mx-auto space-y-8">

        {/* ── Header ── */}
        <div className="animate-slide-up flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
            <p className="text-sm text-gray-400 mt-0.5">Your property analysis reports</p>
          </div>
          <Link
            to="/upload"
            className="px-4 py-2 rounded-xl bg-indigo-600 text-white text-sm font-medium
                       hover:bg-indigo-700 active:scale-95 transition-all shadow-sm"
          >
            + New analysis
          </Link>
        </div>

        {/* ── Live scan ── */}
        {scanId && (
          <section className="animate-slide-up" style={{ animationDelay: '60ms' }}>
            <SectionLabel live>In progress</SectionLabel>
            <ScanTracker scanId={scanId} onComplete={handleScanComplete} />
          </section>
        )}

        {/* ── Reports list ── */}
        <section className="animate-slide-up" style={{ animationDelay: '120ms' }}>
          <SectionLabel>Past reports</SectionLabel>

          {loading ? (
            <Skeleton />
          ) : reports.length === 0 ? (
            <div className="animate-fade-in bg-white rounded-2xl border border-dashed border-gray-200 p-12 text-center">
              <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
                <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round"
                    d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                </svg>
              </div>
              <p className="text-gray-500 text-sm font-medium">No reports yet</p>
              <p className="text-gray-400 text-xs mt-1 mb-4">Upload a video walkthrough to get started</p>
              <Link
                to="/upload"
                className="text-indigo-600 text-sm font-medium hover:underline"
              >
                Upload your first video →
              </Link>
            </div>
          ) : (
            <div className="space-y-3">
              {reports.map((r, i) => (
                <ReportCard key={r.id} report={r} index={i} onDelete={handleDelete} />
              ))}
            </div>
          )}
        </section>

      </div>
    </div>
  )
}

function SectionLabel({ children, live }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      {live && <span className="live-dot" />}
      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{children}</h2>
    </div>
  )
}
