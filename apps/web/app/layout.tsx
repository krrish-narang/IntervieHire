import { FloatingControls } from '@/components/FloatingControls';

import './globals.css';

export const metadata = { title: 'IntervieHire', description: 'AI-powered interview platform' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
	return (
		<html lang="en">
			<body>
				<FloatingControls />
				{children}
			</body>
		</html>
	);
}
