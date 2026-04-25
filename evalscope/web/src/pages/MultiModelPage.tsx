import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import ModelsOverview from '@/components/multi/ModelsOverview'
import ModelComparison from '@/components/multi/ModelComparison'
import EmptyState from '@/components/common/EmptyState'

export default function MultiModelPage() {
  const { t } = useLocale()
  const { multiReportList } = useReports()

  if (multiReportList.length === 0) {
    return <EmptyState text={t('sidebar.note')} />
  }

  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-lg font-semibold mb-3">{t('multi.modelsOverview')}</h2>
        <ModelsOverview reports={multiReportList} />
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">{t('multi.modelComparisonDetails')}</h2>
        <ModelComparison />
      </section>
    </div>
  )
}
