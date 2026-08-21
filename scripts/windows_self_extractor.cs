// Copyright (c) KMG. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Reflection;
using System.Text;
using System.Threading;

internal static class SbkChartsSelfExtractor
{
    private const string AppName = "@@APP_NAME@@";
    private const string Version = "@@VERSION@@";
    private const string Target = "@@TARGET@@";
    private const string StateSchema = "@@STATE_SCHEMA@@";
    private const string RuntimeDirectory = "@@RUNTIME_DIRECTORY@@";
    private const string PayloadSha256 = "@@PAYLOAD_SHA256@@";
    private const int LockTimeoutSeconds = @@LOCK_TIMEOUT_SECONDS@@;
    private const int FooterLength = 8;

    private static string RuntimeRoot;
    private static string InstallDirectory;
    private static string StateFile;
    private static string Executable;
    private static string Marker;

    public static int Main(string[] arguments)
    {
        try
        {
            if (!Environment.Is64BitOperatingSystem || !Environment.Is64BitProcess)
                throw new InvalidOperationException("This application requires " + Target);

            RuntimeRoot = Environment.GetEnvironmentVariable("SBK_CHARTS_PORTABLE_ROOT");
            if (String.IsNullOrWhiteSpace(RuntimeRoot))
            {
                string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                if (String.IsNullOrWhiteSpace(local))
                    throw new InvalidOperationException(
                        "LOCALAPPDATA is unavailable; set SBK_CHARTS_PORTABLE_ROOT");
                RuntimeRoot = Path.Combine(local, RuntimeDirectory);
            }

            InstallDirectory = Path.Combine(RuntimeRoot, Version, Target, PayloadSha256);
            StateFile = Path.Combine(RuntimeRoot, "state-" + Target);
            Executable = Path.Combine(InstallDirectory, AppName + ".exe");
            Marker = Path.Combine(InstallDirectory, ".payload.sha256");

            bool created = false;
            bool reused = IsSavedApplicationReady();
            if (!reused)
            {
                Directory.CreateDirectory(RuntimeRoot);
                string mutexName = "Local\\" + AppName.Replace('-', '_') + "_" + Target.Replace('-', '_');
                using (Mutex mutex = new Mutex(false, mutexName))
                {
                    bool acquired;
                    try
                    {
                        acquired = mutex.WaitOne(TimeSpan.FromSeconds(LockTimeoutSeconds));
                    }
                    catch (AbandonedMutexException)
                    {
                        acquired = true;
                    }
                    if (!acquired)
                        throw new TimeoutException("Timed out waiting for portable extraction lock");
                    try
                    {
                        reused = IsSavedApplicationReady();
                        if (!reused)
                        {
                            ExtractAndPublish();
                            created = true;
                        }
                    }
                    finally
                    {
                        mutex.ReleaseMutex();
                    }
                }
            }

            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = Executable;
            start.Arguments = JoinArguments(arguments);
            start.UseShellExecute = false;
            start.WorkingDirectory = Environment.CurrentDirectory;
            start.EnvironmentVariables["SBK_CHARTS_PORTABLE_SELECTION_SOURCE"] =
                created ? "self-extract-created" : "self-extract-cache";
            start.EnvironmentVariables["SBK_CHARTS_PORTABLE_REUSED"] = reused ? "yes" : "no";
            start.EnvironmentVariables["SBK_CHARTS_PORTABLE_CREATED"] = created ? "yes" : "no";
            start.EnvironmentVariables["SBK_CHARTS_PORTABLE_PREFIX"] = InstallDirectory;
            using (Process child = Process.Start(start))
            {
                child.WaitForExit();
                return child.ExitCode;
            }
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(AppName + ": ERROR: " + error.Message);
            return 1;
        }
    }

    private static bool IsSavedApplicationReady()
    {
        if (!File.Exists(Executable) || !File.Exists(Marker) || !File.Exists(StateFile))
            return false;
        if (!String.Equals(File.ReadAllText(Marker).Trim(), PayloadSha256,
            StringComparison.OrdinalIgnoreCase))
            return false;
        Dictionary<string, string> state = new Dictionary<string, string>();
        foreach (string line in File.ReadAllLines(StateFile))
        {
            int separator = line.IndexOf('=');
            if (separator > 0)
                state[line.Substring(0, separator)] = line.Substring(separator + 1);
        }
        return ValueIs(state, "schema", StateSchema)
            && ValueIs(state, "target", Target)
            && ValueIs(state, "version", Version)
            && ValueIs(state, "payload_sha256", PayloadSha256)
            && ValueIs(state, "install_dir", InstallDirectory);
    }

    private static bool ValueIs(Dictionary<string, string> values, string key, string expected)
    {
        string actual;
        return values.TryGetValue(key, out actual)
            && String.Equals(actual, expected, StringComparison.Ordinal);
    }

    private static void ExtractAndPublish()
    {
        string temporary = Path.Combine(RuntimeRoot, ".install-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temporary);
        try
        {
            string archive = Path.Combine(temporary, "payload.zip");
            CopyAndVerifyPayload(archive);
            string content = Path.Combine(temporary, "content");
            ZipFile.ExtractToDirectory(archive, content);
            if (!File.Exists(Path.Combine(content, AppName + ".exe")))
                throw new InvalidDataException("Extracted application executable is missing");
            File.WriteAllText(Path.Combine(content, ".payload.sha256"), PayloadSha256 + "\n");
            if (Directory.Exists(InstallDirectory))
                Directory.Delete(InstallDirectory, true);
            Directory.CreateDirectory(Path.GetDirectoryName(InstallDirectory));
            Directory.Move(content, InstallDirectory);
            WriteStateAtomically();
        }
        finally
        {
            if (Directory.Exists(temporary))
                Directory.Delete(temporary, true);
        }
    }

    private static void CopyAndVerifyPayload(string destination)
    {
        string self = Assembly.GetExecutingAssembly().Location;
        using (FileStream source = File.OpenRead(self))
        {
            if (source.Length <= FooterLength)
                throw new InvalidDataException("Embedded payload footer is missing");
            source.Seek(-FooterLength, SeekOrigin.End);
            byte[] lengthBytes = new byte[FooterLength];
            ReadExactly(source, lengthBytes, lengthBytes.Length);
            long payloadLength = BitConverter.ToInt64(lengthBytes, 0);
            long payloadOffset = source.Length - FooterLength - payloadLength;
            if (payloadLength <= 0 || payloadOffset < 0)
                throw new InvalidDataException("Embedded payload length is invalid");
            source.Seek(payloadOffset, SeekOrigin.Begin);
            using (FileStream output = File.Create(destination))
            using (SHA256 hasher = SHA256.Create())
            {
                byte[] buffer = new byte[1024 * 1024];
                long remaining = payloadLength;
                while (remaining > 0)
                {
                    int requested = (int)Math.Min(buffer.Length, remaining);
                    int count = source.Read(buffer, 0, requested);
                    if (count <= 0)
                        throw new EndOfStreamException("Embedded payload ended early");
                    output.Write(buffer, 0, count);
                    hasher.TransformBlock(buffer, 0, count, buffer, 0);
                    remaining -= count;
                }
                hasher.TransformFinalBlock(new byte[0], 0, 0);
                string actual = BitConverter.ToString(hasher.Hash).Replace("-", "").ToLowerInvariant();
                if (!String.Equals(actual, PayloadSha256, StringComparison.Ordinal))
                    throw new InvalidDataException("Embedded payload failed SHA-256 verification");
            }
        }
    }

    private static void ReadExactly(Stream source, byte[] destination, int length)
    {
        int offset = 0;
        while (offset < length)
        {
            int count = source.Read(destination, offset, length - offset);
            if (count <= 0)
                throw new EndOfStreamException("Embedded payload footer ended early");
            offset += count;
        }
    }

    private static void WriteStateAtomically()
    {
        string temporary = Path.Combine(RuntimeRoot, ".state-" + Process.GetCurrentProcess().Id + ".tmp");
        File.WriteAllLines(temporary, new[]
        {
            "schema=" + StateSchema,
            "target=" + Target,
            "version=" + Version,
            "payload_sha256=" + PayloadSha256,
            "install_dir=" + InstallDirectory,
        });
        if (File.Exists(StateFile))
            File.Replace(temporary, StateFile, null);
        else
            File.Move(temporary, StateFile);
    }

    private static string JoinArguments(string[] arguments)
    {
        StringBuilder result = new StringBuilder();
        foreach (string argument in arguments)
        {
            if (result.Length > 0)
                result.Append(' ');
            result.Append(QuoteArgument(argument));
        }
        return result.ToString();
    }

    private static string QuoteArgument(string argument)
    {
        StringBuilder quoted = new StringBuilder("\"");
        int backslashes = 0;
        foreach (char character in argument)
        {
            if (character == '\\')
            {
                backslashes++;
            }
            else if (character == '"')
            {
                quoted.Append('\\', backslashes * 2 + 1);
                quoted.Append('"');
                backslashes = 0;
            }
            else
            {
                quoted.Append('\\', backslashes);
                backslashes = 0;
                quoted.Append(character);
            }
        }
        quoted.Append('\\', backslashes * 2);
        quoted.Append('"');
        return quoted.ToString();
    }
}
