import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Loader2, Image as ImageIcon, SendHorizonal, FileText } from "lucide-react";
import { motion } from "framer-motion";

export default function RagProjectSimpleUI() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Replace this with your real backend endpoint
  const API_URL = "http://127.0.0.1:8000/ask";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setAnswer("");
    setImageUrl("");

    if (!question.trim()) {
      setError("Please enter a question first.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: question }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response from the model.");
      }

      const data = await response.json();

      // Expected response shape:
      // {
      //   answer: "text answer here",
      //   image_url: "http://..." or "/images/example.png"
      // }
      setAnswer(data.answer || "No answer returned.");
      setImageUrl(data.image_url || "");
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="mx-auto max-w-5xl space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <Card className="rounded-2xl border-0 shadow-sm">
            <CardHeader>
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <CardTitle className="text-2xl font-bold">Ophthalmology RAG Demo</CardTitle>
                  <p className="mt-2 text-sm text-slate-600">
                    Ask a question about your document collection and get a response with text and image output.
                  </p>
                </div>
                <Badge className="w-fit rounded-full px-3 py-1 text-sm">Simple UI Prototype</Badge>
              </div>
            </CardHeader>
          </Card>
        </motion.div>

        <div className="grid gap-6 lg:grid-cols-2">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <Card className="h-full rounded-2xl border-0 shadow-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <SendHorizonal className="h-5 w-5" />
                  Ask a Question
                </CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <Textarea
                    placeholder="Example: What is the primary goal of glaucoma treatment?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    className="min-h-[180px] rounded-xl"
                  />

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-700">Backend endpoint</label>
                    <Input value={API_URL} readOnly className="rounded-xl bg-slate-100" />
                  </div>

                  {error && (
                    <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                      {error}
                    </div>
                  )}

                  <Button type="submit" disabled={loading} className="w-full rounded-xl py-6 text-base">
                    {loading ? (
                      <span className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Generating response...
                      </span>
                    ) : (
                      "Submit Question"
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
          >
            <Card className="h-full rounded-2xl border-0 shadow-sm">
              <CardHeader>
                <CardTitle className="text-lg">Model Output</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="rounded-2xl border bg-white p-4">
                  <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-700">
                    <FileText className="h-4 w-4" />
                    Text Answer
                  </div>
                  <div className="min-h-[140px] whitespace-pre-wrap text-sm leading-6 text-slate-700">
                    {answer || "The generated answer will appear here."}
                  </div>
                </div>

                <div className="rounded-2xl border bg-white p-4">
                  <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-700">
                    <ImageIcon className="h-4 w-4" />
                    Retrieved / Generated Image
                  </div>

                  {imageUrl ? (
                    <img
                      src={imageUrl}
                      alt="Model returned visual"
                      className="max-h-[360px] w-full rounded-xl border object-contain"
                    />
                  ) : (
                    <div className="flex min-h-[220px] items-center justify-center rounded-xl border border-dashed text-sm text-slate-500">
                      Image output will appear here.
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
        >
          <Card className="rounded-2xl border-0 shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg">Expected Backend Response</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="overflow-x-auto rounded-2xl bg-slate-950 p-4 text-sm text-slate-100">
{`{
  "answer": "Glaucoma treatment primarily aims to lower intraocular pressure...",
  "image_url": "/sample-images/glaucoma-diagram.png"
}`}
              </pre>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
