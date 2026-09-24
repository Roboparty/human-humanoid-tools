import type { UploadFile } from "./api";

function readDirectory(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  return new Promise((resolve, reject) => {
    const entries: FileSystemEntry[] = [];
    const readBatch = () => {
      reader.readEntries((batch) => {
        if (!batch.length) resolve(entries);
        else {
          entries.push(...batch);
          readBatch();
        }
      }, reject);
    };
    readBatch();
  });
}

async function walkEntry(
  entry: FileSystemEntry,
  output: UploadFile[],
  prefix = "",
): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => {
      (entry as FileSystemFileEntry).file(resolve, reject);
    });
    const upload = file as UploadFile;
    upload._relpath = `${prefix}${upload.name}`;
    output.push(upload);
    return;
  }
  if (!entry.isDirectory) return;
  const children = await readDirectory(
    (entry as FileSystemDirectoryEntry).createReader(),
  );
  await Promise.all(
    children.map((child) => walkEntry(child, output, `${prefix}${entry.name}/`)),
  );
}

/** Recursively collect a browser drop while preserving paths needed by FastAPI. */
export async function collectDroppedFiles(
  dataTransfer: DataTransfer | null,
): Promise<UploadFile[]> {
  const files: UploadFile[] = [];
  const items = dataTransfer?.items;
  if (items?.length) {
    const entries: FileSystemEntry[] = [];
    const looseFiles: UploadFile[] = [];
    for (const item of items) {
      const entry = item.webkitGetAsEntry?.();
      if (entry) entries.push(entry);
      else {
        const file = item.getAsFile?.() as UploadFile | null;
        if (file) looseFiles.push(file);
      }
    }
    if (entries.length) await Promise.all(entries.map((entry) => walkEntry(entry, files)));
    for (const file of looseFiles) {
      file._relpath ||= file.webkitRelativePath || file.name;
      files.push(file);
    }
    if (files.length) return files;
  }

  for (const candidate of dataTransfer?.files ?? []) {
    const file = candidate as UploadFile;
    const looksLikeDirectory =
      file.size === 0 && !file.type && !/\.[^/.]+$/.test(file.name);
    if (looksLikeDirectory) continue;
    file._relpath ||= file.webkitRelativePath || file.name;
    files.push(file);
  }
  return files;
}
