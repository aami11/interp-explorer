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

  const getColor = (value: number) => {
    const intensity = Math.min(Math.abs(value) / 1.5, 1)
    return `rgb(${Math.round(59 + intensity * 150)}, ${Math.round(130 - intensity * 80)}, ${Math.round(246 - intensity * 150)})`
  }

  return (
    <div style={{ maxWidth: 700, margin: "0 auto", padding: 20 }}>
      <h1>Interp Explorer</h1>

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          placeholder="Enter a sentence..."
          style={{ flex: 1, padding: 8, fontSize: 16 }}
        />
        <button onClick={analyze} style={{ padding: "8px 16px", fontSize: 16 }}>
          Analyze
        </button>
      </div>

      {loading && <p>Loading...</p>}

      {layers.length > 0 && (
        <>
          <div style={{ marginBottom: 12 }}>
            <button
              onClick={() => setMetric("l2")}
              style={{ fontWeight: metric === "l2" ? "bold" : "normal", marginRight: 8 }}
            >
              L2 Distance
            </button>
            <button
              onClick={() => setMetric("cosine")}
              style={{ fontWeight: metric === "cosine" ? "bold" : "normal" }}
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
                    fill={selectedLayer === entry.layer ? "#f59e0b" : getColor(entry[metric])}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {selectedLayer !== null && (
            <div style={{ marginTop: 16, padding: 12, background: "#f5f5f5", borderRadius: 8 }}>
              <h3>Layer {selectedLayer} Details</h3>
              <p>L2 Distance: {layers[selectedLayer]?.l2}</p>
              <p>Cosine Similarity: {layers[selectedLayer]?.cosine}</p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default App