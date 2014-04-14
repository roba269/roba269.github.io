---
layout: post
title: Leveldb code review part3 - Compaction
---

Let's see the details of compaction. Compaction can be triggered automatically or manually. Let's focus on automatic trigger now.

The compaction starts from function `DBImpl::BackgroundCompaction()`. Pseudo code:

  BackgroundCompaction() {
    if there exists immutable memtable then
      compact imm memtable to Level 0 SSTFile and return
    // we ignore manual compaction
    pick a suitable compaction c;
    DoCompactionWork(c);
    Cleanup;
  }

How to pick a suitable compaction in `VersionSet::PickCompaction()`?

There are member variables like `compaction_score_`, `compaction_level_` in `Version`, which is set at the end of `VersionSet::Finalize()` function. I.e., once a new version is generated, we will check if some level is too large, if so, we mark it as "to be compacted". 

And once `PickCompaction()` is called, we pick one file in the `compaction_level_` of current version, and pick all the overlapped files in the next upper level or in the same level (if level == 0). And also we keep `compact_pointer_[level]` in `VersionSet`, that is, the largest key of previously compacted file of each level. And we pick next file whose key is larger in the next round, make the compaction of each level like "round robin". 

We simply checked the trivial compaction, that is actually move file upper one level. then The actual work is done in function `DBImpl::DoCompactionWork(Compaction *compact)`.

We get an iterator from the compaction spec, which is used for enumerate all k/v in order. And if same key with different seq number appears adjacently, we need to decide if we can drop one (I haven't read the logic carefully), and output the result to a new file using `compact->builder`.

One performance issue here is single threaded. There is quite suitable for the leveldb's original purpose - Chrome embedded db - but might not good enough for server usage. One main contribution of rocksdb is to take advantage of multi-threaded compaction instead. And of course, multi-threaded compaction is much more complicated, hopefully I have a chance to analyze it later.

