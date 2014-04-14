---
layout: post
title: Leveldb code review part2 - How is data stored and the concept of version
---

How is data stored
------------------

Basically, there are two parts of data, in memory and on-disk. 

When write a key into db, we append this information into log file (for recovery), and update it in the memory (called memtable, essentially a skip-list). And if the memtable is large enough, it's dumped into disk, as a SSTable. (Let's put aside the fact that the SSTables can be compacted in background.) 

And when reading, we check the memtable at first, if the key exists, just return it. If not, check the SSTables from newest to oldest. Obviously we can always get the latest value with this approach.

Please note that it's a over-simplified description, I'll add more details later.

Concept of version
------------------

Before diving into the Get/Put details, let's see some important concepts: snapshot / sequence number / versions / VersionEdit. 

These concepts are not quite easy to understand. At the beginning, I thought snapshot and version are same, but it's not. 

1) Snapshot and seq number are working together: essentially, after one operation (insert/update/delete), the seq number will be increased. And snapshot is just a wrapper of seq number, which give a read-only view of the current states of db. I.e., every future operation whose seq number is larger than the snapshot's number will not reflected in this snapshot.

To implement it, we use "internal key" (userkey + seq number + op type) as the key in memtable and SSTable. So we can keep multiple version of userkey in the storage, and fetch the correct version - latest but no later than the snapshot. And we can discard the versions earlier than all the live snapshots.

2) Version / VersionEdit is related with the compaction. Since compaction means adding and removing files, we need log this change for failure recovery. VersionEdit is the log, logging the update based on current version. Once we decide to do compaction, we put the update information into VersionEdit (in memory), generate the physical file, then dump VersionEdit to a new MANIFEST file, and make the CURRENT file point to it.

Question: why do we need VersionSet (multiple versions) in DBImpl? I think we only need to keep the most recent two versions.

