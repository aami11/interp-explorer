import { useState } from "react"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts"

function App() {
  const [inputText, setInputText] = useState("")
  const [layers, setLayers] = useState([])
  const [loading, setLoading] = useState(false)
  const [metric, setMetric] = useState("l2")
  const [selectedLayer, setSelectedLayer] = useState(null)
  const [neurons, setNeurons] = useState([])
  const [neuronLoading, setNeuronLoading] = useState(false)

  const analyze = async () => {
    setLoading(true)
    setSelectedLayer(null)
    setNeurons([])
    const res = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText })
    })
    const data = await res.json()
    setLayers(data.layers)
    setLoading(false)
  }

  const selectLayer = async (layerIdx: number) => {
    setSelectedLayer(layerIdx)
    setNeuronLoading(true)
    const res = await fetch(`http://localhost:8000/neurons/${layerIdx}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText })
    })
    const data = await res.json()
    setNeurons(data.neurons)
    setNeuronLoading(false)
  }

  const getBarColor = (value: number) => {
    const intensity = Math.min(Math.abs(value) / 1.5, 1)
    return `rgb(${Math.round(59 + intensity * 150)}, ${Math.round(130 - intensity * 80)}, ${Math.round(246 - intensity * 150)})`
  }

  const getHeatColor = (value: number, metric: string) => {
    if (metric === "l2") {
      const intensity = Math.min(value / 1.5, 1)
      return `rgb(${Math.round(255 * intensity)}, ${Math.round(255 * (1 - intensity * 0.7))}, ${Math.round(255 * (1 - intensity))})`
    } else if (metric === "cosine") {
      const normalized = (value + 1) / 2
      return `rgb(${Math.round(255 * (1 - normalized))}, ${Math.round(255 * normalized * 0.8)}, ${Math.round(255 * (1 - normalized * 0.5))})`
    } else {
      const intensity = Math.min(value, 1)
      return `rgb(${Math.round(76 * intensity + 220 * (1 - intensity))}, ${Math.round(175 * intensity + 220 * (1 - intensity))}, ${Math.round(80 * intensity + 220 * (1 - intensity))})`
    }
  }

  const getNeuronColor = (value: number) => {
    const intensity = Math.min(value / 0.3, 1)
    return `rgb(${Math.round(234 * intensity + 147 * (1 - intensity))}, ${Math.round(88 * intensity + 197 * (1 - intensity))}, ${Math.round(12 * intensity + 253 * (1 - intensity))})`
  }

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: 20, fontFamily: "system-ui" }}>
      <h1 style={{ textAlign: "center" }}>Interp Explorer</h1>
      <p style={{ textAlign: "center", color: "#666", marginTop: -8 }}>
        Compare activations between TinyStories-1M and TinyStories-3M
      </p>

      <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        <input
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          placeholder="Enter a sentence..."
          style={{ flex: 1, padding: 10, fontSize: 16, borderRadius: 6, border: "1px solid #ccc" }}
        />
        <button
          onClick={analyze}
          style={{ padding: "10px 20px", fontSize: 16, borderRadius: 6, background: "#3b82f6", color: "white", border: "none", cursor: "pointer" }}
        >
          Analyze
        </button>
      </div>

      {loading && <p style={{ textAlign: "center" }}>Analyzing...</p>}

      {layers.length > 0 && (
        <>
          <h2 style={{ fontSize: 18, marginBottom: 8 }}>Activation Heatmap</h2>
          <table style={{ width: "100%", borderCollapse: "collapse", marginBottom: 24 }}>
            <thead>
              <tr>
                <th style={{ padding: 8, textAlign: "left", borderBottom: "2px solid #ddd" }}>Layer</th>
                <th style={{ padding: 8, textAlign: "center", borderBottom: "2px solid #ddd" }}>L2 Distance</th>
                <th style={{ padding: 8, textAlign: "center", borderBottom: "2px solid #ddd" }}>Cosine Similarity</th>
              <th style={{ padding: 8, textAlign: "center", borderBottom: "2px solid #ddd" }}>CKA</th>
            </tr>
            </thead>
            <tbody>
              {layers.map((entry) => (
                <tr
                  key={entry.layer}
                  onClick={() => selectLayer(entry.layer)}
                  style={{ cursor: "pointer", outline: selectedLayer === entry.layer ? "2px solid #3b82f6" : "none" }}
                >
                  <td style={{ padding: 8, fontWeight: 500 }}>Layer {entry.layer}</td>
                  <td style={{
                    padding: 8, textAlign: "center",
                    background: getHeatColor(entry.l2, "l2"),
                    borderRadius: 4, color: "black", fontWeight: 600
                  }}>
                    {entry.l2}
                  </td>
                  <td style={{
                    padding: 8, textAlign: "center",
                    background: getHeatColor(entry.cosine, "cosine"),
                    borderRadius: 4, color: "black", fontWeight: 600
                  }}>
                    {entry.cosine}
                  </td>
                  <td style={{
                    padding: 8, textAlign: "center",
                    background: getHeatColor(entry.cka, "cka"),
                    borderRadius: 4, color: "black", fontWeight: 600
                  }}>
                    {entry.cka}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <h2 style={{ fontSize: 18, marginBottom: 8 }}>Layer Comparison</h2>
          <div style={{ marginBottom: 12 }}>
            <button
              onClick={() => setMetric("l2")}
              style={{
                padding: "6px 14px", marginRight: 8, borderRadius: 4, border: "1px solid #ccc",
                background: metric === "l2" ? "#3b82f6" : "white",
                color: metric === "l2" ? "white" : "black", cursor: "pointer"
              }}
            >
              L2 Distance
            </button>
            <button
              onClick={() => setMetric("cosine")}
              style={{
                padding: "6px 14px", borderRadius: 4, border: "1px solid #ccc",
                background: metric === "cosine" ? "#3b82f6" : "white",
                color: metric === "cosine" ? "white" : "black", cursor: "pointer"
              }}
            >
              Cosine Similarity
            </button>
            <button
              onClick={() => setMetric("cka")}
              style={{
                padding: "6px 14px", marginLeft: 8, borderRadius: 4, border: "1px solid #ccc",
                background: metric === "cka" ? "#3b82f6" : "white",
                color: metric === "cka" ? "white" : "black", cursor: "pointer"
              }}
            >
              CKA
            </button>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={layers} margin={{ left: 20 }}>
              <XAxis dataKey="layer" label={{ value: "Layer", position: "insideBottom", offset: -2 }} />
              <YAxis
                domain={metric === "l2" ? [0, "auto"] : metric === "cosine" ? ["auto", "auto"] : [0, 1]}
                label={{ value: metric === "l2" ? "L2 Distance" : metric === "cosine" ? "Cosine Similarity" : "CKA", angle: -90, position: "insideLeft", offset: -5, style: { textAnchor: "middle" } }}
              />
              <Tooltip />
              <Bar dataKey={metric} onClick={(data) => selectLayer(data.layer)}>
                {layers.map((entry, index) => (
                  <Cell
                    key={index}
                    cursor="pointer"
                    fill={selectedLayer === entry.layer ? "#f59e0b" : getBarColor(entry[metric])}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Neuron Drill-Down */}
          {selectedLayer !== null && (
            <div style={{ marginTop: 24, padding: 16, background: "#f8fafc", borderRadius: 8, border: "1px solid #e2e8f0" }}>
              <h3 style={{ marginTop: 0 }}>Layer {selectedLayer} — Neuron Breakdown</h3>
              <p style={{ color: "#666", fontSize: 14 }}>
                L2: <strong>{layers[selectedLayer]?.l2}</strong> | Cosine: <strong>{layers[selectedLayer]?.cosine}</strong> | CKA: <strong>{layers[selectedLayer]?.cka}</strong>
              </p>

              {neuronLoading && <p>Loading neurons...</p>}

              {neurons.length > 0 && (
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={neurons}>
                    <XAxis dataKey="neuron" label={{ value: "Neuron", position: "insideBottom", offset: -2 }} />
                    <YAxis label={{ value: "L2 Distance", angle: -90, position: "insideLeft" }} />
                    <Tooltip />
                    <Bar dataKey="l2">
                      {neurons.map((entry, index) => (
                        <Cell key={index} fill={getNeuronColor(entry.l2)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default App