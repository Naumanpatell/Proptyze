import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getReport } from '../services/api'

/* ── Shared constants (mirrors Dashboard) ─────────────────────── */

const GRADE_RING = {
  A: '#22c55e', B: '#84cc16', C: '#eab308', D: '#ef4444',
}
const GRADE_BADGE = {
  A: 'bg-green-100 text-green-700 ring-green-300',
  B: 'bg-lime-100  text-lime-700  ring-lime-300',
  C: 'bg-yellow-100 text-yellow-700 ring-yellow-300',
  D: 'bg-red-100   text-red-700   ring-red-300',
}
const GRADE_ACCENT = {
  A: '#bbf7d0', B: '#d9f99d', C: '#fef08a', D: '#fecaca',
}

const CLASS_META = {
  mould:             { label: 'Mould',             category: 'condition' },
  wall_crack:        { label: 'Wall Crack',         category: 'condition' },
  damp:              { label: 'Damp',               category: 'condition' },
  broken_fixture:    { label: 'Broken Fixture',     category: 'condition' },
  peeling_paint:     { label: 'Peeling Paint',      category: 'condition' },
  weak_entry:        { label: 'Weak Entry Point',   category: 'security'  },
  fence_gap:         { label: 'Fence Gap',          category: 'security'  },
  camera_blind_spot: { label: 'Camera Blind Spot',  category: 'security'  },
}

const SEVERITY = {
  mould: 'high', wall_crack: 'high', damp: 'high', weak_entry: 'high',
  broken_fixture: 'medium', fence_gap: 'medium', camera_blind_spot: 'medium',
  peeling_paint: 'low',
}
const SEVERITY_BADGE = {
  high:   'bg-red-50    text-red-600   ring-red-200',
  medium: 'bg-yellow-50 text-yellow-600 ring-yellow-200',
  low:    'bg-gray-50   text-gray-500  ring-gray-200',
}

/* ── Section label — identical to Dashboard ───────────────────── */
function SectionLabel({ children }) {
  return (
    <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
      {children}
    </h2>
  )
}

/* ── Score gauge ──────────────────────────────────────────────── */
function ScoreGauge({ score, grade }) {
  const radius = 48
  const circ   = 2 * Math.PI * radius
  const offset = circ * (1 - score / 100)
  const color  = GRADE_RING[grade] || GRADE_RING.D

  return (
    <div className="flex flex-col items-center gap-2 flex-shrink-0">
      <div className="relative w-32 h-32 flex items-center justify-center">
        <svg width="128" height="128" viewBox="0 0 128 128" className="rotate-[-90deg]">
          <circle cx="64" cy="64" r={radius} fill="none" stroke="#e5e7eb" strokeWidth="9" />
          <circle
            cx="64" cy="64" r={radius}
            fill="none" stroke={color} strokeWidth="9"
            strokeDasharray={circ} strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 1s cubic-bezier(0.16,1,0.3,1)' }}
          />
        </svg>
        <div className="absolute flex flex-col items-center leading-none">
          <span className="text-3xl font-extrabold text-gray-900 animate-score-count"
                style={{ animationDelay: '200ms' }}>
            {score}
          </span>
          <span className="text-[10px] text-gray-400 mt-0.5">/ 100</span>
        </div>
      </div>
      <span className={`animate-pop-in text-xs font-semibold px-2.5 py-1 rounded-full ring-1 ${GRADE_BADGE[grade] || GRADE_BADGE.D}`}
            style={{ animationDelay: '300ms' }}>
        Grade {grade}
      </span>
    </div>
  )
}

/* ── Detection row — styled like a Dashboard report card ──────── */
function DetectionRow({ row, index }) {
  const sev = SEVERITY[row.cls] || 'medium'
  return (
    <div
      className="animate-slide-up flex items-center justify-between
                 bg-white rounded-xl border border-gray-200 px-4 py-3
                 hover:shadow-md hover:-translate-y-0.5 transition-all duration-200"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      <div className="flex items-center gap-3 min-w-0">
        {/* Category dot */}
        <span className={`flex-shrink-0 w-2 h-2 rounded-full ${
          row.category === 'security' ? 'bg-orange-400' : 'bg-indigo-400'
        }`} />
        <div className="min-w-0">
          <p className="text-sm font-medium text-gray-800">{row.label}</p>
          <p className="text-xs text-gray-400">
            {row.count} frame{row.count !== 1 ? 's' : ''} · avg&nbsp;
            {(row.avgConf * 100).toFixed(0)}% confidence
          </p>
        </div>
      </div>
      <span className={`flex-shrink-0 animate-pop-in text-xs font-medium px-2 py-0.5 rounded-full ring-1 ${SEVERITY_BADGE[sev]}`}
            style={{ animationDelay: `${index * 50 + 100}ms` }}>
        {sev}
      </span>
    </div>
  )
}

/* ── Main page ────────────────────────────────────────────────── */
export default function ReportPage() {
  const { id } = useParams()
  const [report, setReport]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState('')

  useEffect(() => {
    getReport(id)
      .then(({ data }) => setReport(data))
      .catch(() => setError('Report not found or could not be loaded.'))
      .finally(() => setLoading(false))
  }, [id])

  /* ── Loading ── */
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin" />
      </div>
    )
  }

  /* ── Error ── */
  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center gap-3">
        <p className="text-sm text-gray-500">{error}</p>
        <Link to="/dashboard" className="text-indigo-600 hover:underline text-sm font-medium">
          ← Back to dashboard
        </Link>
      </div>
    )
  }

  /* ── Build detection rows ── */
  const byClass = {}
  for (const d of report.detections) {
    if (!byClass[d.class]) byClass[d.class] = { count: 0, confidences: [], model: d.model }
    byClass[d.class].count++
    byClass[d.class].confidences.push(d.confidence)
  }
  const detectionRows = Object.entries(byClass).map(([cls, data]) => ({
    cls,
    count:    data.count,
    avgConf:  data.confidences.reduce((a, b) => a + b, 0) / data.confidences.length,
    label:    CLASS_META[cls]?.label    ?? cls.replace(/_/g, ' '),
    category: CLASS_META[cls]?.category ?? data.model,
  }))
  const conditionRows = detectionRows.filter(r => r.category === 'condition')
  const securityRows  = detectionRows.filter(r => r.category === 'security')

  const accentColor = GRADE_ACCENT[report.grade] || GRADE_ACCENT.D

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-12">
      <div className="max-w-2xl mx-auto space-y-8">

        {/* ── Header — same layout as Dashboard ── */}
        <div className="animate-slide-up flex items-center justify-between">
          <div>
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-1.5 text-xs font-medium text-gray-400
                         hover:text-gray-600 transition-colors mb-1.5"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
              Dashboard
            </Link>
            <h1 className="text-2xl font-bold text-gray-900">Inspection Report</h1>
            <p className="text-sm text-gray-400 mt-0.5 truncate max-w-sm">
              {report.filename || 'Property analysis'}
            </p>
          </div>
          <span className="text-xs text-gray-400">
            {new Date(report.created_at).toLocaleDateString('en-GB', {
              day: 'numeric', month: 'short', year: 'numeric',
            })}
          </span>
        </div>

        {/* ── Score card ── */}
        <section className="animate-slide-up" style={{ animationDelay: '60ms' }}>
          <SectionLabel>Overall score</SectionLabel>
          <div
            className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 flex items-center gap-6"
            style={{ borderTopColor: accentColor, borderTopWidth: 3 }}
          >
            <ScoreGauge score={report.score} grade={report.grade} />
            <div className="flex-1 min-w-0 space-y-1.5">
              <p className="text-sm text-gray-600 leading-relaxed">
                {scoreBlurb(report.score, report.grade)}
              </p>
            </div>
          </div>
        </section>

        {/* ── AI Summary ── */}
        {report.ai_summary && (
          <section className="animate-slide-up" style={{ animationDelay: '120ms' }}>
            <div className="flex items-center justify-between mb-3">
              <SectionLabel>AI summary</SectionLabel>
              <span className="text-[10px] text-gray-400 font-medium uppercase tracking-wider mb-3">
                Powered by Claude
              </span>
            </div>
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5
                            hover:shadow-md transition-shadow duration-200">
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 w-7 h-7 rounded-lg bg-indigo-50 flex items-center justify-center mt-0.5">
                  <svg className="w-4 h-4 text-indigo-500" fill="none" viewBox="0 0 24 24"
                       stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-600 leading-relaxed">{report.ai_summary}</p>
              </div>
            </div>
          </section>
        )}

        {/* ── Detections ── */}
        <section className="animate-slide-up" style={{ animationDelay: '180ms' }}>
          <SectionLabel>
            {detectionRows.length > 0 ? 'Issues detected' : 'No issues detected'}
          </SectionLabel>

          {detectionRows.length === 0 ? (
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-8 text-center">
              <div className="w-10 h-10 mx-auto mb-3 rounded-full bg-green-50 flex items-center justify-center">
                <svg className="w-5 h-5 text-green-500" fill="none" viewBox="0 0 24 24"
                     stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="text-sm font-medium text-gray-700">Clean inspection</p>
              <p className="text-xs text-gray-400 mt-1">
                No defects or security concerns were detected in this walkthrough.
              </p>
            </div>
          ) : (
            <div className="space-y-5">
              {conditionRows.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-2 h-2 rounded-full bg-indigo-400" />
                    <span className="text-xs text-gray-400 font-medium">Condition</span>
                  </div>
                  <div className="space-y-2">
                    {conditionRows.map((row, i) => (
                      <DetectionRow key={row.cls} row={row} index={i} />
                    ))}
                  </div>
                </div>
              )}
              {securityRows.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-2 h-2 rounded-full bg-orange-400" />
                    <span className="text-xs text-gray-400 font-medium">Security</span>
                  </div>
                  <div className="space-y-2">
                    {securityRows.map((row, i) => (
                      <DetectionRow key={row.cls} row={row} index={conditionRows.length + i} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        {/* ── Footer note ── */}
        <p className="animate-fade-in text-xs text-center text-gray-400 pb-4"
           style={{ animationDelay: '300ms' }}>
          Score calculated from detected defects, severity weighting, and video lighting quality.
          A professional in-person survey is recommended before making any property decisions.
        </p>

      </div>
    </div>
  )
}

function scoreBlurb(score, grade) {
  if (grade === 'A') return `This property scored ${score}/100 — an excellent result indicating minimal issues and good overall condition.`
  if (grade === 'B') return `This property scored ${score}/100 — a good result with only minor concerns that are worth monitoring.`
  if (grade === 'C') return `This property scored ${score}/100 — some notable issues were found that should be addressed before purchase or let.`
  return `This property scored ${score}/100 — significant issues were detected. A thorough professional survey is strongly recommended.`
}
