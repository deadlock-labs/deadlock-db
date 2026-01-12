// Package web provides embedded HTML templates for the web UI.
package web

import "embed"

//go:embed templates/*
var Templates embed.FS
