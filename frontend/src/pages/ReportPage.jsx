import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getReport } from '../services/api'

const GRADE_STYLES = {
  A: { ring: '#22c55e', bg: 'bg-green-50',   text: 'text-green-700',  badge: 'bg-green-100 ring-green-300' },
  B: { ring: '#84cc16', bg: 'bg-lime-50',    text: 'text-lime-700',   badge: 'bg-lime-100 ring-lime-300' },
  C: { ring: '#eab308', bg: 'bg-yellow-50',  text: 'text-yellow-700', badge: 'bg-yellow-100 ring-yellow-300' },
  D: { ring: '#ef4444', bg: 'bg-red-50',     text: 'text-red-700',    badge: 'bg-red-100 ring-red-300' },
}

const CLASS_LABELS = {
  mould:              { label: 'Mould',              category: 'condition', icon: '🟤' },
  wall_crack:         { label: 'Wall Crack',          category: 'condition', icon: '🪨' },
  damp:               { label: 'Damp',                category: 'condition', icon: '💧' },
  broken_fixture:     { label: 'Broken Fixture',      category: 'condition', icon: '🔧' },
  peeling_paint:      { label: 'Peeling Paint',       category: 'condition', icon: '🎨' },
  weak_entry:         { label: 'Weak Entry Point',    category: 'security',  icon: '🚪' },
  fence_gap:          { label: 'Fence Gap',           category: 'security',  icon: '🏚' },
  camera_blind_spot:  { label: 'Camera Blind Spot',   category: 'security',  icon: '📷' },
}

const SEVERITY = {
  mould: 'high', wall_crack: 'high', damp: 'high',
  weak_entry: 'high',
  broken_fixture: 'medium', fence_gap: 'medium', camera_blind_spot: 'medium',
  peeling_paint: 'low',
}

const SEVERITY_STYLES = {
  high:   'bg-red-50 text-red-700 ring-1 ring-red-200',
  medium: 'bg-yellow-50 text-yellow-700 ring-1 ring-yellow-200',
  low:    'bg-gray-50 text-gray-500 ring-1 ring-gray-200',
}

function ScoreGauge({ score, grade }) {
  const style = GRADE_STYLES[grade] || GRADE_STYLES.D
  const radius = 52
  const circ = 2 * Math.PI * radius
  const offset = circ * (1 - score / 100)

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative w-36 h-36 flex items-center justify-center">
        <svg width="144" height="144" viewBox="0 0 144 144" className="rotate-[-90deg]">
          <circle cx="72" cy="72" r={radius} fill="none" stroke="#e5e7eb" strokeWidth="10" />
          <circle
            cx="72" cy="72" r={radius}
            fill="none" stroke={style.ring} strokeWidth="10"
            strokeDasharray={circ} strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 1s ease' }}
          />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span className="text-4xl font-extrabold text-gray-900">{score}</span>
          <span className="text-xs text-gray-400 -mt-1">/ 100</span>
        </div>
      </div>
      <span className={`text-sm font-semibold px-3 py-1 rounded-full ring-1 ${style.badge} ${style.text}`}>
        Grade {grade}
      </span>
    </div>
  )
}

function DetectionTable({ detections }) {
  // aggregate by class
  const byClass = {}
  for (const d of detections) {
    if (!byClass[d.class]) {
      byClass[d.class] = { count: 0, confidences: [], model: d.model }
    }
    byClass[d.class].count++
    byClass[d.class].confidences.push(d.confidence)
  }

  const rows = Object.entries(byClass).map(([cls, data]) => ({
    cls,
    ...data,
    avgConf: data.confidences.reduce((a, b) => a + b, 0) / data.confidences.length,
    info: CLASS_LABELS[cls] || { label: cls, category: data.model, icon: '⚠' },
    severity: SEVERITY[cls] || 'medium',
  }))

  const condition = rows.filter(r => r.info.category === 'condition')
  const security  = rows.filter(r => r.info.category === 'security')

  const Section = ({ title, items }) => {
    if (!items.length) return null
    return (
      <div>
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">{title}</h3>
        <div className="space-y-2">
          {items.map(row => (
            <div key={row.cls} className="flex items-center justify-between bg-gray-50 rounded-xl px-4 py-3">
              <div className="flex items-center gap-3">
                <span className="text-lg">{row.info.icon}</span>
                <div>
                  <p className="text-sm font-medium text-gray-800">{row.info.label}</p>
                  <p className="text-xs text-gray-400">{row.count} frame{row.count !== 1 ? 's' : ''} · avg {(row.avgConf * 100).toFixed(0)}% confidence</p>
                </div>
              </div>
              <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${SEVERITY_STYLES[row.severity]}`}>
                {row.severity}
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <Section title="Condition issues" items={condition} />
      <Section title="Security concerns" items={security} />
    </div>
  )
}

export default function ReportPage() {
  const { id } = useParams()
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    getReport(id)
      .then(({ data }) => setReport(data))
      .catch(() => setError('Report not found or could not be loaded.'))
      .finally(() => setLoading(false))
  }, [id])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-indigo-300 border-t-indigo-600 rounded-full animate-spin" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center gap-4">
        <p className="text-gray-500">{error}</p>
        <Link to="/dashboard" className="text-indigo-600 hover:underline text-sm">← Back to dashboard</Link>
      </div>
    )
  }

  const style = GRADE_STYLES[report.grade] || GRADE_STYLES.D
  const hasDetections = report.detections && report.detections.length > 0

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-12">
      <div className="max-w-2xl mx-auto space-y-6">

        {/* Back link */}
        <Link to="/dashboard" className="inline-flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-600 transition-colors">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
          Dashboard
        </Link>

        {/* Hero card */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
          <div className={`${style.bg} px-6 py-8 flex flex-col sm:flex-row items-center gap-6`}>
            <ScoreGauge score={report.score} grade={report.grade} />
            <div className="text-center sm:text-left">
              <h1 className="text-xl font-bold text-gray-900">Property Inspection Report</h1>
              <p className="text-sm text-gray-500 mt-1 truncate max-w-sm">{report.filename || 'Property analysis'}</p>
              <p className="text-xs text-gray-400 mt-1">
                {new Date(report.created_at).toLocaleDateString('en-GB', {
                  day: 'numeric', month: 'long', year: 'numeric',
                })}
              </p>
            </div>
          </div>
        </div>

        {/* AI Summary */}
        {report.ai_summary && (
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6">
            <div className="flex items-center gap-2 mb-3">
              <svg className="w-4 h-4 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              <h2 className="text-sm font-semibold text-gray-700">AI Inspection Summary</h2>
              <span className="text-xs text-gray-400 ml-auto">Powered by Claude</span>
            </div>
            <p className="text-sm text-gray-600 leading-relaxed whitespace-pre-line">{report.ai_summary}</p>
          </div>
        )}

        {/* Detections */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">
            {hasDetections ? 'Issues Detected' : 'No Issues Detected'}
          </h2>
          {hasDetections ? (
            <DetectionTable detections={report.detections} />
          ) : (
            <p className="text-sm text-gray-400">
              No defects or security concerns were detected in this property walkthrough.
            </p>
          )}
        </div>

        {/* Score breakdown note */}
        <p className="text-xs text-center text-gray-400 pb-4">
          Score calculated from detected defects, severity weighting, and video lighting quality.
          A professional in-person survey is recommended before making any property decisions.
        </p>

      </div>
    </div>
  )
}
