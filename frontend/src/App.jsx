import { useState } from "react";

export default function App() {

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [loading, setLoading] = useState(false);

  const API_URL = "http://127.0.0.1:8000/ask";

  const askQuestion = async () => {

    if (!question.trim()) return;

    setLoading(true);

    const res = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ prompt: question })
    });

    const data = await res.json();

    setAnswer(data.answer);
    setImageUrl(data.image_url);
    setLoading(false);
  };

  return (
    <div style={{padding:"40px", fontFamily:"Arial"}}>

      <h1>Ophthalmology RAG Demo</h1>

      <textarea
        rows="5"
        style={{width:"500px"}}
        placeholder="Ask a question..."
        value={question}
        onChange={(e)=>setQuestion(e.target.value)}
      />

      <br/><br/>

      <button onClick={askQuestion}>
        {loading ? "Generating..." : "Submit"}
      </button>

      <hr/>

      <h3>Answer</h3>

      <p>{answer}</p>

      {imageUrl && (
        <img
          src={imageUrl}
          style={{maxWidth:"400px"}}
        />
      )}

    </div>
  );
}
