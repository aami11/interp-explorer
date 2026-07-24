import { useState } from "react"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts"

function App() {
  const [inputText, setInputText] = useState("")
  const [layers, setLayers] = useState([])
  const [loading, setLoading] = useState(false)
  const [metric, setMetric] = useState("l2")
  const [selectedLayer, setSelectedLayer] = useState(null)

  const analyze = async () => {
    setLoading(true)
    setSelectedLayer(null)
    const res = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText })
    })
    const data = await res.json()
    setLayers(data.layers)
    setLoading(false)
  }

  const getBarColor = (value: number) => {
    const intensity = Math.min(Math.abs(value) / 1.5, 1)
    return `rgb(${Math.round(59 + intensity * 150)}, ${Math.round(130 - intensity * 80)}, ${Math.round(246 - intensity * 150)})`
  }

  const getHeatColor = (value: number, metric: string) => {
    if (metric === "l2") {
      const intensity = Math.min(value / 1.5, 1)
      return `rgb(${Math.round(255 * intensity)}, ${Math.round(255 * (1 - intensity * 0.7))}, ${Math.round(255 * (1 - intensity))})` 
    } else {
      const normalized = (value + 1) / 2
      return `rgb(${Math.round(255 * (1 - normalized))}, ${Math.round(255 * normalized * 0.8)}, ${Math.round(255 * (1 - normalized * 0.5))})`
    }
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
          {/* Heatmap Grid */}
          <h2 style={{ fontSize: 18, marginBottom: 8 }}>Activation Heatmap</h2>
          <table style={{ width: "100%", borderCollapse: "collapse", marginBottom: 24 }}>
            <thead>
              <tr>
                <th style={{ padding: 8, textAlign: "left", borderBottom: "2px solid #ddd" }}>Layer</th>
                <th style={{ padding: 8, textAlign: "center", borderBottom: "2px solid #ddd" }}>L2 Distance</th>
                <th style={{ padding: 8, textAlign: "center", borderBottom: "2px solid #ddd" }}>Cosine Similarity</th>
              </tr>
            </thead>
            <tbody>
              {layers.map((entry) => (
                <tr
                  key={entry.layer}
                  onClick={() => setSelectedLayer(entry.layer)}
                  style={{ cursor: "pointer", outline: selectedLayer === entry.layer ? "2px solid #3b82f6" : "none" }}
                >
                  <td style={{ padding: 8, fontWeight: 500 }}>Layer {entry.layer}</td>
                  <td style={{
                    padding: 8,
                    textAlign: "center",
                    background: getHeatColor(entry.l2, "l2"),
                    color: "black",
                    borderRadius: 4
                  }}>
                    {entry.l2}
                  </td>
                  <td style={{
                    padding: 8,
                    textAlign: "center",
                    background: getHeatColor(entry.cosine, "cosine"),
                    color: "black",
                    borderRadius: 4
                  }}>
                    {entry.cosine}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Bar Chart */}
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
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={layers}>
              <XAxis dataKey="layer" label={{ value: "Layer", position: "insideBottom", offset: -2 }} />
              <YAxis label={{ value: metric === "l2" ? "L2 Distance" : "Cosine Similarity", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Bar dataKey={metric} onClick={(data) => setSelectedLayer(data.layer)}>
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

          {/* Detail Panel */}
          {selectedLayer !== null && (
            <div style={{ marginTop: 16, padding: 16, background: "#f8fafc", borderRadius: 8, border: "1px solid #e2e8f0" }}>
              <h3 style={{ marginTop: 0 }}>Layer {selectedLayer} Details</h3>
              <p>L2 Distance: <strong>{layers[selectedLayer]?.l2}</strong></p>
              <p>Cosine Similarity: <strong>{layers[selectedLayer]?.cosine}</strong></p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default App