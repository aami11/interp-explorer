import { useState } from "react"

function App() {
  const [inputText, setInputText] = useState("")
  const [layers, setLayers] = useState([])
  const [loading, setLoading] = useState(false)

  const analyze = async () => {
    setLoading(true)
    const res = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText })
    })
    const data = await res.json()
    setLayers(data.layers)
    setLoading(false)
  }

  return (
    <div>
      <h1>Interp Explorer</h1>
      <input
        value={inputText}
        onChange={e => setInputText(e.target.value)}
        placeholder="Enter a sentence..."
      />
      <button onClick={analyze}>Analyze</button>
      {loading && <p>Loading...</p>}
      {layers.map(layer => (
        <p key={layer.layer}>
          Layer {layer.layer} — L2: {layer.l2} | Cosine: {layer.cosine} | CKA: {layer.cka}
        </p>
      ))}
    </div>
  )
}

export default App