/****************************************************************************
** Meta object code from reading C++ file 'csv2gpr_worker.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "include/sensor_stream_ros/ui/csv2gpr_worker.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'csv2gpr_worker.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_CSV2GPRWorker_t {
    QByteArrayData data[30];
    char stringdata0[255];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CSV2GPRWorker_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CSV2GPRWorker_t qt_meta_stringdata_CSV2GPRWorker = {
    {
QT_MOC_LITERAL(0, 0, 13), // "CSV2GPRWorker"
QT_MOC_LITERAL(1, 14, 17), // "predict1dFinished"
QT_MOC_LITERAL(2, 32, 0), // ""
QT_MOC_LITERAL(3, 33, 15), // "gpr::Prediction"
QT_MOC_LITERAL(4, 49, 4), // "pred"
QT_MOC_LITERAL(5, 54, 9), // "hpChanged"
QT_MOC_LITERAL(6, 64, 11), // "lmlComplete"
QT_MOC_LITERAL(7, 76, 3), // "min"
QT_MOC_LITERAL(8, 80, 3), // "max"
QT_MOC_LITERAL(9, 84, 10), // "dataLoaded"
QT_MOC_LITERAL(10, 95, 4), // "size"
QT_MOC_LITERAL(11, 100, 8), // "readFile"
QT_MOC_LITERAL(12, 109, 5), // "fname"
QT_MOC_LITERAL(13, 115, 9), // "predict1d"
QT_MOC_LITERAL(14, 125, 5), // "x_min"
QT_MOC_LITERAL(15, 131, 5), // "x_max"
QT_MOC_LITERAL(16, 137, 6), // "size_t"
QT_MOC_LITERAL(17, 144, 9), // "divisions"
QT_MOC_LITERAL(18, 154, 5), // "y_val"
QT_MOC_LITERAL(19, 160, 6), // "genLML"
QT_MOC_LITERAL(20, 167, 12), // "QCustomPlot*"
QT_MOC_LITERAL(21, 180, 10), // "customPlot"
QT_MOC_LITERAL(22, 191, 12), // "QCPColorMap*"
QT_MOC_LITERAL(23, 204, 8), // "colorMap"
QT_MOC_LITERAL(24, 213, 5), // "y_min"
QT_MOC_LITERAL(25, 219, 5), // "y_max"
QT_MOC_LITERAL(26, 225, 2), // "nx"
QT_MOC_LITERAL(27, 228, 2), // "ny"
QT_MOC_LITERAL(28, 231, 12), // "setBlockSize"
QT_MOC_LITERAL(29, 244, 10) // "block_size"

    },
    "CSV2GPRWorker\0predict1dFinished\0\0"
    "gpr::Prediction\0pred\0hpChanged\0"
    "lmlComplete\0min\0max\0dataLoaded\0size\0"
    "readFile\0fname\0predict1d\0x_min\0x_max\0"
    "size_t\0divisions\0y_val\0genLML\0"
    "QCustomPlot*\0customPlot\0QCPColorMap*\0"
    "colorMap\0y_min\0y_max\0nx\0ny\0setBlockSize\0"
    "block_size"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CSV2GPRWorker[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   54,    2, 0x06 /* Public */,
       5,    0,   57,    2, 0x06 /* Public */,
       6,    2,   58,    2, 0x06 /* Public */,
       9,    1,   63,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      11,    1,   66,    2, 0x0a /* Public */,
      13,    4,   69,    2, 0x0a /* Public */,
      19,    8,   78,    2, 0x0a /* Public */,
      28,    1,   95,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double, QMetaType::Double,    7,    8,
    QMetaType::Void, QMetaType::Int,   10,

 // slots: parameters
    QMetaType::Void, QMetaType::QString,   12,
    QMetaType::Void, QMetaType::Float, QMetaType::Float, 0x80000000 | 16, QMetaType::Float,   14,   15,   17,   18,
    QMetaType::Void, 0x80000000 | 20, 0x80000000 | 22, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, 0x80000000 | 16, 0x80000000 | 16,   21,   23,   14,   15,   24,   25,   26,   27,
    QMetaType::Void, QMetaType::Int,   29,

       0        // eod
};

void CSV2GPRWorker::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<CSV2GPRWorker *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->predict1dFinished((*reinterpret_cast< gpr::Prediction(*)>(_a[1]))); break;
        case 1: _t->hpChanged(); break;
        case 2: _t->lmlComplete((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 3: _t->dataLoaded((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->readFile((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 5: _t->predict1d((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2])),(*reinterpret_cast< size_t(*)>(_a[3])),(*reinterpret_cast< float(*)>(_a[4]))); break;
        case 6: _t->genLML((*reinterpret_cast< QCustomPlot*(*)>(_a[1])),(*reinterpret_cast< QCPColorMap*(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6])),(*reinterpret_cast< size_t(*)>(_a[7])),(*reinterpret_cast< size_t(*)>(_a[8]))); break;
        case 7: _t->setBlockSize((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 6:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QCPColorMap* >(); break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QCustomPlot* >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (CSV2GPRWorker::*)(gpr::Prediction );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&CSV2GPRWorker::predict1dFinished)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (CSV2GPRWorker::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&CSV2GPRWorker::hpChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (CSV2GPRWorker::*)(double , double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&CSV2GPRWorker::lmlComplete)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (CSV2GPRWorker::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&CSV2GPRWorker::dataLoaded)) {
                *result = 3;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject CSV2GPRWorker::staticMetaObject = { {
    &QObject::staticMetaObject,
    qt_meta_stringdata_CSV2GPRWorker.data,
    qt_meta_data_CSV2GPRWorker,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *CSV2GPRWorker::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CSV2GPRWorker::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CSV2GPRWorker.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int CSV2GPRWorker::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void CSV2GPRWorker::predict1dFinished(gpr::Prediction _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void CSV2GPRWorker::hpChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void CSV2GPRWorker::lmlComplete(double _t1, double _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void CSV2GPRWorker::dataLoaded(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
