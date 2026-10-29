# v51 固定容量物理时间信用预检

[TARGET.md](TARGET.md)给出唯一目标及边界；`credit.py`实现，`run.py`读取服务器v43/v44/v46/v49/v50已有数据并写入本目录。`test_credit.py`六项定向测试已通过。

[decision.md](decision.md)记录结论；`summary.json`为唯一下载的结果汇总（31,444字节），细节数组留在服务器。`decision.json`保留分线路对比与后续闸门。

high提交被既有调度器状态锁阻塞，随后在node005直接只读计算一次。`execution.json`保留首次因汇总超限而退出的完整记录；`export_execution.json`记录从现成结果精简导出，未重算数组。`staging_failure.json`单列上传失败，均未产生反馈或仿真开销。原版本和结果未改。
