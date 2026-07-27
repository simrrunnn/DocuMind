import { useEffect, useRef, useState } from "react";
import { deleteDocument, documentFileUrl, listDocuments, uploadDocument } from "../api";

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Sidebar({ onDocumentUploaded }) {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const refresh = async () => {
    try {
      setDocuments(await listDocuments());
    } catch {
      setError("Could not load documents.");
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    try {
      await uploadDocument(file);
      await refresh();
      onDocumentUploaded?.();
    } catch {
      setError("Upload failed. Please try again.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDelete = async (docId) => {
    setDeletingId(docId);
    setError(null);
    try {
      await deleteDocument(docId);
      await refresh();
    } catch {
      setError("Delete failed. Please try again.");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <aside className="sidebar">
      <h2>Documents</h2>
      <label className="upload-button">
        {uploading ? "Uploading..." : "Upload a PDF"}
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          onChange={handleFileChange}
          disabled={uploading}
          hidden
        />
      </label>
      {error && <p className="sidebar-error">{error}</p>}
      <ul className="document-list">
        {documents.map((doc) => (
          <li key={doc.id}>
            <a href={documentFileUrl(doc.id)} target="_blank" rel="noreferrer">
              {doc.filename}
            </a>
            <span className="document-size">{formatSize(doc.size)}</span>
            <button
              type="button"
              className="document-delete"
              aria-label={`Delete ${doc.filename}`}
              onClick={() => handleDelete(doc.id)}
              disabled={deletingId === doc.id}
            >
              ×
            </button>
          </li>
        ))}
        {documents.length === 0 && <li className="document-empty">No documents yet</li>}
      </ul>
    </aside>
  );
}
