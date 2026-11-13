"""Rebind only the diagnosed chipset controller; requires local sudo."""
import os
import time
from pathlib import Path

CONTROLLER = '0000:0e:00.0'
device = Path('/sys/bus/pci/devices') / CONTROLLER
driver = Path('/sys/bus/pci/drivers/xhci_hcd')


def main():
    if os.geteuid() != 0:
        raise SystemExit('请使用 sudo python3 运行此脚本。')
    if (device / 'vendor').read_text().strip() != '0x1022' or (device / 'device').read_text().strip() != '0x43fc':
        raise SystemExit('控制器型号与诊断记录不符，未修改。')
    if (device / 'driver').resolve() != driver.resolve():
        raise SystemExit('控制器未绑定预期驱动，未修改。')
    allowed = {('2b03', 'f681'), ('2b03', 'f682'), ('048d', '5711'), ('0bda', '8922')}
    for usb in Path('/sys/bus/usb/devices').iterdir():
        if CONTROLLER not in str(usb.resolve()) or not (usb / 'idVendor').exists() or usb.name.startswith('usb'):
            continue
        identity = ((usb / 'idVendor').read_text().strip(), (usb / 'idProduct').read_text().strip())
        if identity not in allowed:
            raise SystemExit(f'该控制器出现未预期设备 {usb.name} {identity}，未修改。')
    print(f'重载 {CONTROLLER}；蓝牙及灯控会短暂断开。', flush=True)
    try:
        (driver / 'unbind').write_text(CONTROLLER)
        time.sleep(2)
    finally:
        if not (device / 'driver').exists():
            (driver / 'bind').write_text(CONTROLLER)
    time.sleep(4)
    if not (device / 'driver').exists():
        raise SystemExit('驱动未恢复，请检查系统日志。')
    video = []
    for usb in Path('/sys/bus/usb/devices').iterdir():
        if (usb / 'idVendor').exists() and (usb / 'idVendor').read_text().strip() == '2b03':
            product = (usb / 'idProduct').read_text().strip()
            print('ZED 接口:', usb.name, product, '速率:', (usb / 'speed').read_text().strip(), 'Mbps')
            if product == 'f682':
                video.append(usb.name)
    print('控制器重载完成；' + ('ZED 视频接口已枚举，仍需采帧验证。' if video else 'ZED 视频接口仍未枚举。'))


if __name__ == '__main__':
    main()
