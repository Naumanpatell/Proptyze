import { useState, useRef, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadFiles } from '../services/api'

const VIDEO_EXTS = ['.mp4', '.mov', '.avi']
const IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
const MAX_MB = 500
const MAX_IMAGES = 20

const isVideo = (file) => VIDEO_EXTS.some(ext => file.name.toLowerCase().endsWith(ext))
const isImage = (file) => IMAGE_EXTS.some(ext => file.name.toLowerCase().endsWith(ext))
const fmtMB   = (bytes) => (bytes / (1024 * 1024)).toFixed(1)

function DropZone({ dragging, onDragOver, onDragLeave, onDrop, onClick, label, hint, children }) {
  return (
    <div
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onClick}
      className={`
        relative flex flex-col items-center justify-center gap-3
        border-2 border-dashed rounded-2xl p-12 cursor-pointer
        transition-colors duration-200
        ${dragging
          ? 'border-indigo-500 bg-indigo-50'
          : 'border-gray-300 bg-white hover:border-indigo-400 hover:bg-indigo-50/40'}
      `}
    >
      <svg
        className={`w-14 h-14 ${dragging ? 'text-indigo-500' : 'text-gray-400'} transition-colors`}
        fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.2}
      >
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M3 16.5A4.5 4.5 0 007.5 21h9a4.5 4.5 0 000-9h-.273A6 6 0 105.818 7.5" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 12v6m0-6l-2 2m2-2l2 2" />
      </svg>
      <p className="text-base font-medium text-gray-700">
        {dragging ? 'Drop it here' : label}
      </p>
      <p className="text-sm text-gray-400">{hint}</p>
      <div className="flex items-center gap-3 w-full max-w-xs">
        <span className="flex-1 h-px bg-gray-200" />
        <span className="text-xs text-gray-400 uppercase tracking-wide">or</span>
        <span className="flex-1 h-px bg-gray-200" />
      </div>
      <button
        type="button"
        className="px-5 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium
                   hover:bg-indigo-700 active:scale-95 transition-all"
        onClick={(e) => { e.stopPropagation(); onClick() }}
      >
        Browse files
      </button>
      {children}
    </div>
  )
}

export default function UploadPage() {
  const navigate     = useNavigate()
  const videoRef     = useRef(null)
  const imageRef     = useRef(null)
  const extraImgRef  = useRef(null)

  const [mode, setMode]       = useState('video') // 'video' | 'photos'
  const [video, setVideo]     = useState(null)
  const [images, setImages]   = useState([])       // [{ file, preview }]
  const [dragging, setDragging] = useState(false)
  const [error, setError]     = useState('')
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress]   = useState(0)
  const [success, setSuccess]     = useState(null)

  // Revoke object URLs on unmount
  useEffect(() => () => images.forEach(i => URL.revokeObjectURL(i.preview)), [])

  useEffect(() => {
    if (!success) return
    const t = setTimeout(() => navigate(`/dashboard?scan_id=${success.scanId}`), 2500)
    return () => clearTimeout(t)
  }, [success, navigate])

  const switchMode = (m) => {
    setMode(m)
    setVideo(null)
    images.forEach(i => URL.revokeObjectURL(i.preview))
    setImages([])
    setError('')
    setProgress(0)
  }

  const pickVideo = useCallback((file) => {
    setError('')
    if (!isVideo(file)) { setError('Please select a video file (MP4, MOV, or AVI).'); return }
    if (file.size > MAX_MB * 1024 * 1024) { setError(`File exceeds ${MAX_MB} MB limit.`); return }
    setVideo(file)
  }, [])

  const addImages = useCallback((incoming) => {
    setError('')
    const valid = []
    for (const f of incoming) {
      if (!isImage(f)) { setError(`"${f.name}" is not a supported image type.`); continue }
      if (f.size > MAX_MB * 1024 * 1024) { setError(`"${f.name}" exceeds ${MAX_MB} MB.`); continue }
      valid.push({ file: f, preview: URL.createObjectURL(f) })
    }
    setImages(prev => {
      const combined = [...prev, ...valid]
      if (combined.length > MAX_IMAGES) {
        setError(`Max ${MAX_IMAGES} photos per upload. Extra photos were removed.`)
        valid.slice(MAX_IMAGES - prev.length).forEach(i => URL.revokeObjectURL(i.preview))
        return combined.slice(0, MAX_IMAGES)
      }
      return combined
    })
  }, [])

  const removeImage = (idx) => {
    setImages(prev => {
      URL.revokeObjectURL(prev[idx].preview)
      return prev.filter((_, i) => i !== idx)
    })
  }

  const clearImages = () => {
    images.forEach(i => URL.revokeObjectURL(i.preview))
    setImages([])
  }

  // Drag & drop
  const onDragOver  = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const files = Array.from(e.dataTransfer.files)
    if (mode === 'video') {
      if (files[0]) pickVideo(files[0])
    } else {
      addImages(files)
    }
  }

  const handleUpload = async () => {
    setUploading(true)
    setError('')
    try {
      const allFiles = mode === 'video'
        ? [video, ...images.map(i => i.file)].filter(Boolean)
        : images.map(i => i.file)
      const { data } = await uploadFiles(allFiles, setProgress)
      setUploading(false)
      const label = mode === 'video'
        ? images.length > 0 ? `${video.name} + ${images.length} photo${images.length > 1 ? 's' : ''}` : video.name
        : `${images.length} photo${images.length > 1 ? 's' : ''}`
      setSuccess({ scanId: data.scan_id, filename: label })
    } catch (err) {
      const msg = err.response?.data?.detail || err.response?.data?.message || 'Upload failed. Please try again.'
      setError(msg)
      setUploading(false)
    }
  }

  const reset = () => {
    setVideo(null)
    clearImages()
    setError('')
    setProgress(0)
  }

  const canUpload = mode === 'video' ? !!video : images.length > 0

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4 py-12">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-gray-900">Analyse Your Property</h1>
        <p className="mt-2 text-gray-500">
          Upload a video walkthrough or photos — we'll detect defects, score the property, and generate a report.
        </p>
      </div>

      <div className="w-full max-w-xl space-y-4">

        {/* Mode switcher */}
        {!uploading && !success && (
          <div className="flex bg-white border border-gray-200 rounded-xl p-1 gap-1">
            <ModeTab active={mode === 'video'} onClick={() => switchMode('video')} icon={<VideoIcon />} label="Video" />
            <ModeTab active={mode === 'photos'} onClick={() => switchMode('photos')} icon={<PhotoIcon />} label="Photos" />
          </div>
        )}

        {/* ── VIDEO MODE ── */}
        {mode === 'video' && !uploading && !success && (
          <>
            {!video ? (
              <DropZone
                dragging={dragging}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onClick={() => videoRef.current?.click()}
                label="Drag & drop your video here"
                hint={`MP4, MOV, or AVI · max ${MAX_MB} MB`}
              >
                <input ref={videoRef} type="file" accept={VIDEO_EXTS.join(',')} className="hidden"
                  onChange={(e) => { const f = e.target.files[0]; if (f) pickVideo(f); e.target.value = '' }} />
              </DropZone>
            ) : (
              /* Selected video card */
              <div className="bg-white border border-gray-200 rounded-2xl p-5 flex items-center gap-4 shadow-sm">
                <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-indigo-100 flex items-center justify-center">
                  <VideoIcon className="w-6 h-6 text-indigo-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-gray-800 truncate">{video.name}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{fmtMB(video.size)} MB</p>
                </div>
                <button onClick={() => setVideo(null)} className="text-gray-400 hover:text-red-500 transition-colors" aria-label="Remove video">
                  <XIcon />
                </button>
              </div>
            )}

            {/* Optional: add supporting photos (only after video is picked) */}
            {video && (
              <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-700">Supporting photos <span className="text-gray-400 font-normal">(optional)</span></p>
                    <p className="text-xs text-gray-400">Add up to {MAX_IMAGES} photos of specific issues</p>
                  </div>
                  <button
                    onClick={() => extraImgRef.current?.click()}
                    className="px-3 py-1.5 rounded-lg border border-gray-300 text-xs font-medium text-gray-600 hover:bg-gray-50 transition-colors"
                  >
                    + Add photos
                  </button>
                  <input ref={extraImgRef} type="file" accept={IMAGE_EXTS.join(',')} multiple className="hidden"
                    onChange={(e) => { addImages(Array.from(e.target.files)); e.target.value = '' }} />
                </div>

                {images.length > 0 && (
                  <>
                    <ImageGrid images={images} onRemove={removeImage} />
                    <button onClick={clearImages} className="text-xs text-red-400 hover:text-red-600">Remove all photos</button>
                  </>
                )}
              </div>
            )}
          </>
        )}

        {/* ── PHOTOS MODE ── */}
        {mode === 'photos' && !uploading && !success && (
          <>
            <DropZone
              dragging={dragging}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => imageRef.current?.click()}
              label={images.length > 0 ? 'Drop more photos here' : 'Drag & drop your photos here'}
              hint={`JPG, PNG, WEBP · up to ${MAX_IMAGES} photos · max ${MAX_MB} MB each`}
            >
              <input ref={imageRef} type="file" accept={IMAGE_EXTS.join(',')} multiple className="hidden"
                onChange={(e) => { addImages(Array.from(e.target.files)); e.target.value = '' }} />
            </DropZone>

            {images.length > 0 && (
              <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-sm font-medium text-gray-700">
                    {images.length} / {MAX_IMAGES} photos selected
                  </p>
                  <button onClick={clearImages} className="text-xs text-red-400 hover:text-red-600">Remove all</button>
                </div>
                <ImageGrid images={images} onRemove={removeImage} />
              </div>
            )}
          </>
        )}

        {/* Upload progress */}
        {uploading && (
          <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 font-medium truncate max-w-[70%]">
                {mode === 'video' ? video?.name : `${images.length} photos`}
              </span>
              <span className="text-indigo-600 font-semibold">{progress}%</span>
            </div>
            <div className="w-full bg-gray-100 rounded-full h-2">
              <div className="bg-indigo-500 h-2 rounded-full transition-all duration-300" style={{ width: `${progress}%` }} />
            </div>
            <p className="text-xs text-gray-400">Uploading… please don't close this tab.</p>
          </div>
        )}

        {/* Success card */}
        {success && (
          <div className="bg-white border border-green-200 rounded-2xl p-8 shadow-sm flex flex-col items-center gap-4 animate-fade-in">
            <div className="relative flex items-center justify-center w-20 h-20">
              <svg className="absolute inset-0 w-20 h-20 animate-spin-once" viewBox="0 0 80 80">
                <circle cx="40" cy="40" r="36" fill="none" stroke="#22c55e" strokeWidth="4"
                  strokeDasharray="226" strokeDashoffset="0" className="origin-center" />
              </svg>
              <div className="w-14 h-14 rounded-full bg-green-100 flex items-center justify-center">
                <svg className="w-8 h-8 text-green-500 animate-draw-check" fill="none" viewBox="0 0 24 24"
                     stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 13l4 4L19 7" />
                </svg>
              </div>
            </div>
            <div className="text-center">
              <p className="text-lg font-semibold text-gray-900">Uploaded successfully!</p>
              <p className="text-sm text-gray-500 mt-1 truncate max-w-xs">{success.filename}</p>
              <p className="text-xs text-gray-400 mt-3">Taking you to your dashboard…</p>
            </div>
            <div className="flex gap-1.5">
              {[0, 1, 2].map((i) => (
                <span key={i} className="w-2 h-2 rounded-full bg-green-400 animate-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }} />
              ))}
            </div>
          </div>
        )}

        {/* Error banner */}
        {error && (
          <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-600">
            <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 15.75h.007v.008H12v-.008z" />
            </svg>
            {error}
          </div>
        )}

        {/* Action buttons */}
        {canUpload && !uploading && !success && (
          <div className="flex gap-3">
            <button onClick={reset}
              className="flex-1 py-3 rounded-xl border border-gray-300 text-sm font-medium
                         text-gray-600 hover:bg-gray-100 transition-colors">
              {mode === 'video' ? 'Choose different file' : 'Clear all'}
            </button>
            <button onClick={handleUpload}
              className="flex-1 py-3 rounded-xl bg-indigo-600 text-white text-sm font-semibold
                         hover:bg-indigo-700 active:scale-95 transition-all shadow-sm">
              Analyse Property
            </button>
          </div>
        )}

      </div>
    </div>
  )
}

function ModeTab({ active, onClick, icon, label }) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all
        ${active ? 'bg-indigo-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'}`}
    >
      {icon}
      {label}
    </button>
  )
}

function ImageGrid({ images, onRemove }) {
  return (
    <div className="grid grid-cols-4 gap-2">
      {images.map((img, i) => (
        <div key={i} className="relative group aspect-square">
          <img src={img.preview} alt="" className="w-full h-full object-cover rounded-lg" />
          <button
            onClick={() => onRemove(i)}
            className="absolute top-1 right-1 w-5 h-5 rounded-full bg-black/60 text-white
                       flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  )
}

function VideoIcon({ className = 'w-5 h-5' }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M15.75 10.5l4.72-2.36A.75.75 0 0121 8.868v6.264a.75.75 0 01-1.03.696L15.75 13.5M4.5 7.5h9a1.5 1.5 0 011.5 1.5v6a1.5 1.5 0 01-1.5 1.5h-9A1.5 1.5 0 013 15V9a1.5 1.5 0 011.5-1.5z" />
    </svg>
  )
}

function PhotoIcon({ className = 'w-5 h-5' }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v13.5a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V9.75zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
    </svg>
  )
}

function XIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}
