---
layout: post
title: Leveldb code review part4 - table cache and iterators
---

Next step...

TableCache
----------

First thing first: why and when is cache needed? Cache is useful when we are trying to read sst files. They are on disk, loading them into memory is expensive, so we want to cache hot files in memory.

The cache is a sharded LRU cache. It contains a group by LRU cache, sharded by hash(key) at first. (Is it really helpful for the performance?) I won't describe the LRU implementation coz it's really standard.

One interesting thing is that the capacity of cache is not defined by the number of elements. Instead, each element has a (may be different) "charge" - if the total charge is larger than cache's capacity, some element will be evicted.

The other interesting thing is, the "deleter" is bind with the elements, when the element is evicted from cache, the deleter function will be called. So it's essentially a dtor, but since we are keeping `(void*)` inside the cache, we have to use the approach to simulate dtor.

Let see some platform specific things. When `TableCache::FindTable()` is called, we will look for the file on disk and open it using `Env::NewRandomAccessFile()`. For `PosixEnv`, we are using `mmap`, which mapping a file into a memory region. (I'm not sure about the performance actually, I remember I heard somebody said mmap is slow in some scenarios.)

Iterator
--------

There are many kinds of iterators implementing the Iterator interface. Though seems complicated, most of the ideas are straight forward.

Let's see a typical one using when we do compaction - `MergingIterator`. It actually contains multiply child iterators, and we pick the next item like a k-way merge. Duplicated key (same user key but different seq number) can be returned one by one, leaving the dedup logic to callers. Among the child iterators, for the tables whose `level > 0` (so no overlap between tables), there is one `TwoLevelIterator` maintaining all the tables. (TwoLevel means table level and item level.) No need to do the k-way merge inside this level.
