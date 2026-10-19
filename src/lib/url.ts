/** Astro 的站点根路径；当前根域名部署时为 `/`。 */
export const BASE = import.meta.env.BASE_URL;

/** 生成站内路径；外部链接与锚点原样返回。 */
export function withBase(path: string): string {
  if (/^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(path) || path.startsWith("#")) {
    return path;
  }
  return `${BASE.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

/** 判断某个导航项是否为当前页面，用于 aria-current。 */
export function isCurrent(pathname: string, href: string): boolean {
  const target = withBase(href);
  return href === "/" ? pathname === target : pathname.startsWith(target);
}
