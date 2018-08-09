function Sign(X : Integer)
   return Integer is
begin
   if X >= 1 then
      return 1;
   elsif X > 0 then
      return -1;
   else
      return 0;
   end if;
end Sign;