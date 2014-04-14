---
layout: post
title: Leveldb code review part1 - High level description
---

Disclaimer: These articles are not guaranteed to be precise/correct, use them at your own risk.

Leveldb is an embedded k/v database from Google, based on the idea of LSM tree. Though it's super important to understand the scope and rationale of an open-source project before reading its code, I won't talk too much high-level things here, because there have been great documents from the authors.

So let us dive into the "middle-level", taking a look at the source code structure:

+ The interface is exposed at `include/leveldb/*.h`. And the most important one is `include/leveldb/db.h`
+ Most of the implementation code is at `db/`. There are some important classes, including `DBImpl`, `Version/VersionSet/VersionEdit`, `LogReader/LogWriter`, `MemTable`, etc.
+ Some platform-dependent code at `port/`, like file IO and multi-thread.
+ The structure of table file at `table/`
+ Some utility functions at `util/`, `helper/`

