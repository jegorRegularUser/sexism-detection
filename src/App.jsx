import { useState } from "react";
import "./App.css";

function App() {
  const [load, setLoad] = useState(false);
  const [text, setText] = useState("");
  const [res, setRes] = useState('');
  const [model, setModel] = useState("local_3");
  const [history, setHistory] = useState([]); // Для хранения истории
  const [showHistory, setShowHistory] = useState(false); // Для отображения истории

  const handleChange = (event) => {
    setText(event.target.value);
  };

  const handleModelChange = (event) => {
    setModel(event.target.value);
  };

  const checkHandler = async () => {
    setLoad(true);
    await fetch("https://sexism-detection.onrender.com/predict/", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, model })
    })
      .then(res => res.json())
      .then(res => {
        const result = res.prediction === 1 ? "Your message is Sexist" : "Your message is Not Sexist";
        setRes(result);

        setHistory((prevHistory) => [
          { text, model, result },
          ...prevHistory,
        ]);
      })
      .catch((e) => {
        setRes('Sorry, try again');
        console.error(e);
      })
      .finally(() => {
        setLoad(false);
      });
  };

  const toggleHistory = () => {
    setShowHistory(!showHistory);
  };

  return (
    <>
      <h1>Let's check you on sexism!</h1>
      <p>Please write message below</p>

      <div className="input">
        <select value={model} onChange={handleModelChange}>
          <option value="local_3">Local Model 3e</option>
          <option value="local_5">Local Model 5e</option>
          <option value="collab_5">Collab Model 5e</option>
          <option value="collab_7">Collab Model 7e</option>
          <option value="2_collab_3">2.0 Collab Model 3e</option>
        </select>

        <textarea
          value={text}
          onChange={handleChange}
          placeholder="Here..."
        ></textarea>
        <button onClick={checkHandler} disabled={!text || load}>
          {load ? "Checking..." : "Check!"}
        </button>
      </div>

      {!load && (res ? <p>{res}</p> : <p></p>)}

      <div className="history-container">
        <button className="history-button" onClick={toggleHistory}>
          {showHistory ? "Hide Past Answers" : "Show Past Answers"}
        </button>

        {showHistory && (
          <ul className="history-list">
            {history.map((item, index) => (
              <li key={index} className="history-item">
                <div className="model">Model: {item.model}</div>
                <div className="query"> {item.text}</div>
                <div className="separator"></div>
                <div className="response"> {item.result}</div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}

export default App;
