# Skill: YouTube Download (Authorized Content Only)

## Skill Name

youtube_download_video

## Description

Downloads a YouTube video only if the content is authorized for download
(e.g., Creative Commons license, public domain, or user-owned content).

This skill must not bypass YouTube protections or Terms of Service.

------------------------------------------------------------------------

## Intent

User wants: - A downloadable copy of a YouTube video - Audio or video
format - Saved locally or to cloud storage

------------------------------------------------------------------------

## Input Schema

``` json
{
  "video_url": "string",
  "format": "mp4 | mp3",
  "quality": "highest | 1080p | 720p | audio_only"
}
```

Example:

``` json
{
  "video_url": "https://www.youtube.com/watch?v=example",
  "format": "mp4",
  "quality": "720p"
}
```

------------------------------------------------------------------------

## Output Schema

``` json
{
  "status": "success | failure",
  "file_path": "string",
  "file_size_mb": "number",
  "duration_seconds": "number",
  "notes": "string (optional)"
}
```

------------------------------------------------------------------------

## Preconditions

-   video_url must contain `/watch?v=`
-   Video must be publicly accessible
-   Video must be authorized for download
-   Internet connection available
-   Sufficient disk space available

------------------------------------------------------------------------

## Execution Plan

1.  Validate URL
    -   Confirm URL matches YouTube watch pattern\
    -   Reject playlist-only links
2.  Fetch Metadata
    -   Retrieve title, duration, license type, availability\
    -   Confirm license is Creative Commons, public domain, or owned by
        user\
    -   If not authorized → return failure
3.  Select Format
    -   If format = mp4 → select video stream\
    -   If format = mp3 → select audio stream\
    -   Apply quality preference if available
4.  Download Stream
    -   Download selected stream\
    -   Save file using sanitized video title\
    -   Monitor progress
5.  Post-Processing
    -   If mp3: extract audio and remove temporary video file
6.  Validate Download
    -   Check file exists\
    -   Check file size \> 0\
    -   Confirm duration matches metadata
7.  Return Structured Output

------------------------------------------------------------------------

## Failure Handling

Unauthorized Content:

``` json
{
  "status": "failure",
  "notes": "Video is not authorized for download."
}
```

Video Not Found:

``` json
{
  "status": "failure",
  "notes": "Video not found or unavailable."
}
```

Network Failure:

``` json
{
  "status": "failure",
  "notes": "Network error during download."
}
```

Insufficient Storage:

``` json
{
  "status": "failure",
  "notes": "Insufficient disk space."
}
```

------------------------------------------------------------------------

## Safety & Compliance

-   Do not bypass DRM
-   Do not extract streams via undocumented APIs
-   Do not automate Premium features
-   Respect YouTube Terms of Service
-   Log authorization checks

------------------------------------------------------------------------

## Observability

Log: - video_url - license type - selected format - download duration -
file size - failure reason (if any)

------------------------------------------------------------------------

## Edge Cases

-   Region-locked content\
-   Age-restricted content\
-   Live streams\
-   Scheduled premieres\
-   Disabled downloads

Return failure for restricted content.

------------------------------------------------------------------------

## Agent Reasoning Pattern

1.  Validate input\
2.  Confirm authorization\
3.  Retrieve metadata\
4.  Select appropriate stream\
5.  Download\
6.  Verify integrity\
7.  Return structured result

------------------------------------------------------------------------

## Summary

This skill enables compliant, controlled downloading of authorized
YouTube content while enforcing validation, licensing checks, and
structured output handling.
