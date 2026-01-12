// Package main provides the command-line interface for deadlock-db.
package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	deadlockdb "github.com/deadlock-labs/deadlock-db"
	"github.com/deadlock-labs/deadlock-db/pkg/api"
)

func main() {
	// Command-line flags
	addr := flag.String("addr", ":8080", "HTTP server address")
	dataDir := flag.String("data", "", "Data directory for persistent storage (empty for in-memory)")
	version := flag.Bool("version", false, "Print version and exit")

	flag.Parse()

	if *version {
		fmt.Printf("deadlock-db version %s\n", deadlockdb.Version())
		os.Exit(0)
	}

	// Create database
	opts := &deadlockdb.Options{
		DataDir: *dataDir,
	}
	db, err := deadlockdb.Open(opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open database: %v\n", err)
		os.Exit(1)
	}

	// Create and start server
	server := api.NewServer(db, *addr)

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		fmt.Println("\nShutting down...")
		if err := db.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing database: %v\n", err)
		}
		os.Exit(0)
	}()

	fmt.Printf("deadlock-db %s starting on %s\n", deadlockdb.Version(), *addr)
	if *dataDir != "" {
		fmt.Printf("Data directory: %s\n", *dataDir)
	} else {
		fmt.Println("Running in memory-only mode")
	}

	if err := server.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}
