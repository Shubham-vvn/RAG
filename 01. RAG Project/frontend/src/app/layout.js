import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Ask My PDF — RAG Grounded Document Intelligence',
  description: 'Upload a PDF and ask questions using Retrieval-Augmented Generation (RAG)',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
